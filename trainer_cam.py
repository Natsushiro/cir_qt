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
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import wandb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
from third_party.open_clip.simple_tokenizer import SimpleTokenizer
from utils import is_master

from torchvision.transforms import ToPILImage, Normalize, Compose
from PIL import Image


def get_loss(model, images, texts, loss_img, loss_txt, args, data_identifier=-1):
    if data_identifier == 1:
        # ImageNet dataset
        image_features, text_features, logit_scale = model(images, texts, extra=True)
    else:
        image_features, text_features, logit_scale = model(images, texts)   
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss

def get_text_features_original(model, token_features, args):
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
    return text_features

def get_text_features(model, token_features, captions, pos):
    text_features = model.encode_text_mask(captions, token_features, pos)
    return text_features

def inverse_transform(tensor):
    denormalize = Normalize(mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
                            std=[1/0.26862954, 1/0.26130258, 1/0.27577711])

    to_pil = ToPILImage()
    inverse_transform = Compose([
        denormalize,
        to_pil,
    ])

    pil_image = inverse_transform(tensor)
    return pil_image

def get_loss_imgtext2text(model, img2text, masking, images, texts, pos, labels, loss_img, loss_txt, args, memory=None):
    masked_images = masking(images, labels)
    for img_id in range(len(masked_images)):
        im = masked_images[img_id]
        im = inverse_transform(im)
        im.save("test/test_{}.png".format(str(img_id)))
    #raise ValueError("Stop")
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        #masked_image_features, _, ids_restore = model.encode_masked_image(images)
        masked_features = model.encode_image(masked_images)
    #token_features = img2text(masked_image_features, ids_restore)
    token_features1 = img2text(image_features)
    token_features2 = img2text(masked_features)
    text_features1 = get_text_features_original(model, token_features1, args)
    text_features2 = get_text_features(model, token_features2, texts, pos)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
    text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features1 = [
            torch.zeros_like(text_features1) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features1, text_features1)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features1 = torch.cat(
            [text_features1]
            + gathered_text_features1[:rank]
            + gathered_text_features1[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features1.t()
        #logits_per_caption = logit_scale * all_caption_features @ all_text_features.t()
        # print(logits_per_image.type())
        # print(ground_truth.type())
        loss_img_val = loss_img(logits_per_image, ground_truth)
        #loss_cap_val = loss_img(logits_per_caption, ground_truth)
        logits_per_text = logits_per_image.t()
        #logits_per_txtcap = logits_per_caption.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
        #loss_txtcap_val = loss_txt(logits_per_txtcap, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image_org = logit_scale * image_features @ text_features1.t()
        loss_img_val_org = loss_img(logits_per_image_org, ground_truth)
        logits_per_text_org = logit_scale * text_features1 @ image_features.t()
        loss_txt_val_org = loss_txt(logits_per_text_org, ground_truth)

        logits_per_image_msk = logit_scale * image_features @ text_features2.t()
        loss_img_val_msk = loss_img(logits_per_image_msk, ground_truth)
        logits_per_text_msk = logit_scale * text_features2 @ image_features.t()
        loss_txt_val_msk = loss_txt(logits_per_text_msk, ground_truth)
    total_loss = (loss_img_val_org + loss_txt_val_org) / 2 + args.alpha * (loss_img_val_msk + loss_txt_val_msk) / 2
    return total_loss

def get_loss_imgtext2text_multi(model, img2text, masking, images, texts, pos, labels, loss_img, loss_txt, args, memory=None):
    masked_images = masking(images, labels) # (num*B, C, H, W)
    num = 5
    #for img_id in range(len(masked_images)):
    #    im = masked_images[img_id]
    #    im = inverse_transform(im)
    #    im.save("test/test_{}.png".format(str(img_id)))
    #raise ValueError("Stop")
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        #masked_image_features, _, ids_restore = model.encode_masked_image(images)
        masked_features = model.encode_image(masked_images) # (num*B, D)
        masked_features = masked_features.view(num, image_features.shape[0], -1) # (num, B, D)
        masked_features = masked_features.mean(dim=0)

    #token_features = img2text(masked_image_features, ids_restore)
    token_features1 = img2text(image_features)
    token_features2 = img2text(masked_features)
    text_features1 = get_text_features_original(model, token_features1, args)
    text_features2 = get_text_features(model, token_features2, texts, pos)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
    text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features1 = [
            torch.zeros_like(text_features1) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features1, text_features1)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features1 = torch.cat(
            [text_features1]
            + gathered_text_features1[:rank]
            + gathered_text_features1[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features1.t()
        #logits_per_caption = logit_scale * all_caption_features @ all_text_features.t()
        # print(logits_per_image.type())
        # print(ground_truth.type())
        loss_img_val = loss_img(logits_per_image, ground_truth)
        #loss_cap_val = loss_img(logits_per_caption, ground_truth)
        logits_per_text = logits_per_image.t()
        #logits_per_txtcap = logits_per_caption.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
        #loss_txtcap_val = loss_txt(logits_per_txtcap, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image_org = logit_scale * image_features @ text_features1.t()
        loss_img_val_org = loss_img(logits_per_image_org, ground_truth)
        logits_per_text_org = logit_scale * text_features1 @ image_features.t()
        loss_txt_val_org = loss_txt(logits_per_text_org, ground_truth)

        logits_per_image_msk = logit_scale * image_features @ text_features2.t()
        loss_img_val_msk = loss_img(logits_per_image_msk, ground_truth)
        logits_per_text_msk = logit_scale * text_features2 @ image_features.t()
        loss_txt_val_msk = loss_txt(logits_per_text_msk, ground_truth)
    total_loss = (loss_img_val_org + loss_txt_val_org) / 2 + args.alpha * (loss_img_val_msk + loss_txt_val_msk) / 2
    return total_loss


def train(model, img2text, masking, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, preprocess=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    data['train'].set_epoch(epoch)
    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        #images, texts, masks = batch[0], batch[1], batch[2]
        images, texts = batch[0], batch[1]
        text = torch.stack([a[0] for a in texts])
        pos = [a[1] for a in texts]
        labels = torch.stack([a[2] for a in texts])
        #if len(batch) == 3 and args.use_debiased_sampler:
        #    data_identifier = torch.unique(batch[2])[0].numpy()
        #else:
        data_identifier = -1
        if args.gpu is not None:
            text = text.cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss_imgtext2text(m, img2text, masking, images, text, pos, labels, loss_img, loss_txt, args, data_identifier)
                scaler.scale(total_loss).backward()
                #for name, param in model.named_parameters():
                #    print(name, ":", param.grad)
                #for name, param in img2text.named_parameters():
                #    print(name, ":", param.grad)   
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss = get_loss_imgtext2text(m, img2text, masking, images, text, pos, labels, loss_img, loss_txt, args, data_identifier) 
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        #m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})