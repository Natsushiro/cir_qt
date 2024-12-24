import json
from pathlib import Path

import PIL
import torch.utils.data
import torchvision
from torch.utils.data import Dataset

from argparse import ArgumentParser
from params import parse_args
from collections import defaultdict

import numpy as np

import os
import time
import logging
from time import gmtime, strftime
from functools import partial
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT
from eval_utils import evaluate_imgnet_retrieval, evaluate_coco, evaluate_fashion, evaluate_cirr, evaluate_cirr_test
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, TargetPad
from tqdm import tqdm
import clip

def encode_with_pseudo_tokens(clip_model, text, pseudo_tokens,
                              num_tokens=1):
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def load_model(args):
    model, _, preprocess_val = load(
            args.model,
            jit=False)
    img2text = IM2TEXT(embed_dim=model.embed_dim, 
                       middle_dim=args.middle_dim, 
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer) 
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
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
            sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
        model.load_state_dict(sd)
        img2text.load_state_dict(sd_img2text)
        logging.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
        )
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, img2text, preprocess_val

def setup_log_save(args):
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

class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path, split,
                 mode, preprocess):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index):
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index):
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

base_path = Path(__file__).absolute().parents[1].absolute()  # Getting the path to the base directory


@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset, clip_model, index_features,
                              index_names, ref_names_list, pseudo_tokens):
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, relative_val_dataset,
                                                                                   ref_names_list, pseudo_tokens)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.cuda()
    predicted_features = predicted_features.cuda()

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }

@torch.no_grad()
def circo_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list,
                                   pseudo_tokens):
    """
    Generates features predictions for the validation set of CIRCO
    """

    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']

        gt_img_ids = np.array(gt_img_ids).T.tolist()
        input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).cuda()
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list

@torch.no_grad()
def extract_image_features(dataset, clip_model, batch_size=64,
                           num_workers=8):
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.cuda()
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names

@torch.no_grad()
def circo_val_retrieval(dataset_path, clip_model, ref_names_list, pseudo_tokens,
                        preprocess):
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    """
    # Extract the index features
    classic_val_dataset = CIRCODataset(dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(dataset_path, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names, ref_names_list,
                                     pseudo_tokens)

@torch.no_grad()
def extract_pseudo_tokens_with_phi(clip_model, phi, dataset):
    """
    Extracts pseudo tokens from a dataset using a CLIP model and a phi model
    """
    data_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=10, pin_memory=False,
                             collate_fn=collate_fn)
    predicted_tokens = []
    names_list = []
    print(f"Extracting tokens using phi model")
    for batch in tqdm(data_loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')
        

        images = images.cuda()
        image_features = clip_model.encode_image(images)

        batch_predicted_tokens = phi(image_features)
        predicted_tokens.append(batch_predicted_tokens.cpu())
        names_list.extend(names)

    predicted_tokens = torch.vstack(predicted_tokens)
    return predicted_tokens, names_list

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    # Log and save params.
    setup_log_save(args)
    # Load trained model
    model, img2text, preprocess_val = load_model(args)
    cudnn.benchmark = True
    cudnn.deterministic = False   
    root_project = os.path.join(get_project_root(), 'data')
    relative_val_dataset = CIRCODataset("../data/CIRCO", 'val', 'relative', preprocess_val)


    pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(model, img2text, relative_val_dataset)
    pseudo_tokens = pseudo_tokens.cuda()

    circo_metrics = circo_val_retrieval("../data/CIRCO", model, ref_names_list, pseudo_tokens,
                                        preprocess_val)

    for k, v in circo_metrics.items():
        print(f"{k} = {v:.2f}")

def main():
    args = parse_args()

    if args.name is None:
        args.name = (f"lr={args.lr}_"
            "wd={args.wd}_"
            "agg={args.aggregate}_"
            "model={args.model}_"
            "batchsize={args.batch_size}_workers={args.workers}")
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
    args.world_size = 1
    main_worker(args.gpu, None, log_queue, args)


if __name__ == '__main__':
    main()