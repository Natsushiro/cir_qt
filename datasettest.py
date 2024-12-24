import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import io

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from open_clip import get_tokenizer
import nltk

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights

def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    #sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes_filename = os.path.join(dir_path, os.path.splitext(os.path.basename(shards))[0]+"_stats.json")
    json_list, _ = expand_urls(sizes_filename)
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(json_list[0]):
        json_files = [json.load(open(f, 'r')) for f in json_list]
        sizes = [int(j["successes"]) for j in json_files]
        total_size = sum(sizes)
        print(total_size)
        #total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()

def mydecoder(data):
    return Image.open(io.BytesIO(data))


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

# the first NN word is replaced by an "*"
tokenizer = get_tokenizer("ViT-L-14")
def process_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = nltk.pos_tag(tokens)
    i = 1
    for w in tokens:
        if w[1] == 'NN':
            pos = i
            break
        i = i+1
    text = tokenizer(text)[0]
    text[pos] = tokenizer("*")[0][1]

    return text

    #return text

def unprocess_text(text):
    return text

def get_wds_dataset(args, preprocess_img, preprocess_val, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        assert args.train_data_upsampling_factors is None,\
            "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode(wds.handle_extension("jpg", mydecoder),
            wds.handle_extension("mask.jpg", mydecoder)),
        wds.rename(image="jpg", text="txt", mask="mask.jpg"),
        wds.map_dict(image=preprocess_img, text=process_text, mask=preprocess_val),
        #wds.map_dict(image=preprocess_img, text=unprocess_text),
        wds.to_tuple("image", "text", "mask"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    data["train"] = get_wds_dataset(
            args, preprocess_train, preprocess_val, is_train=True, epoch=epoch, tokenizer=tokenizer)

    return data

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from third_party.open_clip.scheduler import cosine_lr
from model.clip_mask import _transform, load
from model.model_mask import convert_weights, CLIP, IM2TEXT
from trainer_mask import train
#from webdata import get_data
from params import parse_args
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32
from open_clip import get_tokenizer
import torchvision.transforms as T

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    
    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(
            args.model,
            jit=False)
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        if args.use_prefix:
            model_info['vocab_size'] += 1
            model_info['use_prefix'] = True
        model = CLIP(**model_info)
        convert_weights(model)        
        preprocess_train = _transform(model.visual.input_resolution, is_train=False)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)
    try:
        img2text = IM2TEXT(embed_dim=model.embed_dim, 
                           middle_dim=args.middle_dim, 
                           output_dim=model.token_embedding.weight.shape[1], 
                           n_layer=args.n_layer)
    except:
        img2text = IM2TEXT(embed_dim=1024, output_dim=1024,
        is_normalize=args.normalize_output, is_mlp=args.use_mlp, n_layer=args.n_layer)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)

    for name, param in img2text.named_parameters():
        if "proj" in name or "ln" in name:
            param.requires_grad_(False)
    #for p in img2text.named_parameters():
    #    print(p[0], p[1].requires_grad)

    
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)
    named_parameters = list(img2text.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    
    start_epoch = 0
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    #data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    #print(get_tokenizer(args.model))

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            sd_img2text = checkpoint["state_dict_img2text"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
                sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
            model.load_state_dict(sd)
            img2text.load_state_dict(sd_img2text)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    cudnn.deterministic = False
    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
        train(model, img2text, data, epoch, optimizer, scaler, scheduler, args, writer)
        steps = data["train"].dataloader.num_batches * (epoch + 1)        
        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, "epoch_latest.pt"),
                )

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()


def main():
    args = parse_args()

    # get the name of the experiments
    if args.name is None:
        args.name = (f"lr={args.lr}_"
            "wd={args.wd}_"
            "agg={args.aggregate}_"
            "model={args.model}_"
            "batchsize={args.batch_size}_workers={args.workers}")
        import pdb
        pdb.set_trace
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    # args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()