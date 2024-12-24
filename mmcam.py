import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchattacks.attack import Attack
from torchvision.transforms import InterpolationMode

start_layer = -1

def get_MMCAM(model, images, labels, start_layer=start_layer, upsample=None, get_patch_cam=False):
    # Ref: https://github.com/hila-chefer/Transformer-MM-Explainability

    # Get activations and gradients for each attention map
    batch_size = len(labels)
    #attentions = []
    logits_per_image, _ = model(images, labels)
    #ground_truth = torch.arange(batch_size).long()
    #ground_truth = ground_truth.cuda()
    #loss = logits_per_image.gather(1, ground_truth.unsqueeze(1)).sum()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    #one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    #one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    #one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #loss = torch.sum(one_hot.cuda() * logits_per_image)

    #logits = model(images, attn_out=attentions)
    #loss = logits.gather(1, labels.unsqueeze(1)).sum()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    L = model.visual.positional_embedding.size(0)
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).cuda()
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    patch_cam = R[:, 0, 1:]
    cam = patch_cam.T
    if upsample:
        s = round((L - 1) ** 0.5)
        cam = patch_cam.reshape(batch_size, s, s)
        cam = TF.resize(cam, images.size()[2:], upsample)  # N * H * W
    if get_patch_cam:
        return cam, patch_cam
    return cam

    #grads = torch.autograd.grad(loss, attentions)
    #attentions = [attn.detach() for attn in attentions]

    # Compute CAM
    #bare_model = model.module if hasattr(model, 'module') else model
    #N = images.size(0)
    #L = bare_model.positional_embedding.size(0)
    #R = torch.eye(L, device=images.device).repeat(N, 1, 1)
    #for attn, grad in zip(attentions, grads):
    #    A = (grad * attn).clamp(min=0)                  # (N * Nh) * L * L
    #    A = A.reshape(N, -1, L, L).mean(dim=1)          # N * L * L
    #    R += torch.matmul(A, R)                         # N * L * L
    #patch_cam = R[:, 0, 1:]
    #cam = patch_cam.T                                   # (L-1) * N
    #if upsample:
    #    s = round((L - 1) ** 0.5)
    #    cam = patch_cam.reshape(N, s, s)
    #    cam = TF.resize(cam, images.size()[2:], upsample)  # N * H * W
    #if get_patch_cam:
    #    return cam, patch_cam
    #return cam

class CAMMaskSingleFill(Attack):

    def __init__(self, model, threshold, ctx_mask=False, save_mask=False):
        super().__init__("CAMMaskSingleFill", model)
        self.get_CAM = partial(
                get_MMCAM, upsample=InterpolationMode.NEAREST, get_patch_cam=save_mask)
        self.threshold = threshold
        self.ctx_mask = ctx_mask
        self.save_mask = save_mask

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device) # B, C, H, W
        cam = self.get_CAM(self.model, images, labels) # cam.shape = (L-1) * N or N * H *W
        if self.save_mask:
            cam, patch_cam = cam
            patch_cam = patch_cam / (patch_cam.amax(dim=1, keepdim=True) + 1e-8)
            patch_mask = (patch_cam > self.threshold)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        mask = (cam > self.threshold).unsqueeze(1)
        if self.ctx_mask:
            mask = ~mask
            if self.save_mask:
                patch_mask = ~patch_mask
        idx = torch.randperm(images.size()[0], device=self.device)
        
        images_masked = images * (~mask) + images[idx] * (mask) # shuffled image batch
        if self.save_mask:
            return images_masked, patch_mask
        return images_masked
    
class CAMMaskSingleFill_noise(Attack):

    def __init__(self, model, threshold, ctx_mask=False, save_mask=False):
        super().__init__("CAMMaskSingleFill", model)
        self.get_CAM = partial(
                get_MMCAM, upsample=InterpolationMode.NEAREST, get_patch_cam=save_mask)
        self.threshold = threshold
        self.ctx_mask = ctx_mask
        self.save_mask = save_mask

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device) # B, C, H, W
        B, C, H, W = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
        cam = self.get_CAM(self.model, images, labels) # cam.shape = (L-1) * N or N * H *W
        if self.save_mask:
            cam, patch_cam = cam
            patch_cam = patch_cam / (patch_cam.amax(dim=1, keepdim=True) + 1e-8)
            patch_mask = (patch_cam > self.threshold)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        mask = (cam > self.threshold).unsqueeze(1)
        if self.ctx_mask:
            mask = ~mask
            if self.save_mask:
                patch_mask = ~patch_mask
        idx = torch.randperm(images.size()[0], device=self.device)

        noise_image = torch.rand(B, C, H, W, device=self.device)

        images_masked = images * (~mask) +  noise_image[idx] * (mask) # noise
        if self.save_mask:
            return images_masked, patch_mask
        return images_masked

class CAMMaskSingleFill_MultipleImages(Attack):

    def __init__(self, model, threshold, ctx_mask=False, save_mask=False):
        super().__init__("CAMMaskSingleFill", model)
        self.get_CAM = partial(
                get_MMCAM, upsample=InterpolationMode.NEAREST, get_patch_cam=save_mask)
        self.threshold = threshold
        self.ctx_mask = ctx_mask
        self.save_mask = save_mask

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device) # B, C, H, W
        batch_size, channel, height, width = images.shape
        cam = self.get_CAM(self.model, images, labels) # cam.shape = (L-1) * N or N * H *W
        if self.save_mask:
            cam, patch_cam = cam
            patch_cam = patch_cam / (patch_cam.amax(dim=1, keepdim=True) + 1e-8)
            patch_mask = (patch_cam > self.threshold)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        mask = (cam > self.threshold).unsqueeze(1)
        if self.ctx_mask:
            mask = ~mask
            if self.save_mask:
                patch_mask = ~patch_mask
        #idx = torch.randperm(images.size()[0], device=self.device)
        num = 5
        idx = torch.stack([torch.randperm(images.size()[0]) for i in range(num)]) # generate num groups of ids size: (num, B)
        images = images.repeat(num, 1, 1, 1, 1) # num, B, C, H, W
        for i in range(num):
            images_masked = images[i] * (~mask) + images[i][idx] * (mask) # shuffled image batch. waiting for optimization
        images_masked = images_masked.view(num * batch_size, channel, height, width)
        if self.save_mask:
            return images_masked, patch_mask
        return images_masked

def get_masking(args, **kwargs):
    #return CAMMaskSingleFill_MultipleImages(kwargs["model"], args.cam_threshold, ctx_mask=True)
    return CAMMaskSingleFill(kwargs["model"], args.cam_threshold, ctx_mask=True)
    
    
    
