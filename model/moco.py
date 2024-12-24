# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from third_party.open_clip.clip import tokenize
import openai
from open_clip import get_tokenizer
#from model import IM2TEXT
import concurrent.futures


API_URL = "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"
openai.api_key = API_URL
syst = "You are going to imagine the situation of the two sentences. Use one sentence to only tell the change from the situation in text1 to the situation in text2. Do not mention the word \"text1\" and \"text2\"."
model = "gpt-3.5-turbo-16k"
tokenizer = get_tokenizer("ViT-L-14")

def get_response(prompt):
   response = openai.ChatCompletion.create(model=model, messages=[{"role": "system", "content": syst}, {"role": "user", "content": prompt}])
   return response["choices"][0]["message"]["content"]

def get_text_features(model, token_features):
    text = tokenize("a photo of")
    text = text.cuda(0, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
    return text_features



class DifferentialNet(nn.Module):
    def __init__(self, embed_dim=768, middle_dim=512, output_dim=768, n_layer=2, dropout=0.1):
        super().__init__()
        self.source_linear = nn.Sequential(
            nn.Linear(embed_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, output_dim)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        #diff = torch.cat((source, target), dim=-1)
        diff = target - source
        diff = self.source_linear(diff)
        diff = nn.functional.normalize(diff, dim=-1)
        return diff

class CombineNet(nn.Module):
    def __init__(self, embed_dim=768, middle_dim=512, output_dim=768, n_layer=2, dropout=0.1):
        super().__init__()
        self.img_linear = nn.Sequential(
            nn.Linear(embed_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.txt_linear = nn.Sequential(
            nn.Linear(embed_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.combine_linear = nn.Sequential(
            nn.Linear(2 * middle_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, output_dim)
        )
        self.weight_linear = nn.Sequential(
            nn.Linear(2 * middle_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, output_dim)
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor):
        x1 = self.img_linear(img)
        x2 = self.txt_linear(txt)
        concat = torch.cat((x1, x2), dim=-1)
        combine = self.combine_linear(concat)
        weight = self.weight_linear(concat)
        x1 = img.mul(torch.special.exp2(weight))
        x2 = txt.mul(torch.special.exp2(-weight))
        out = combine + x1 + x2
        out = nn.functional.normalize(out, dim=-1)
        return out


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, clip_model, diff_net, combine_net, args, dim=768, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        
        self.diff_net_q = diff_net()
        # diff_net_k = diff_net()
        self.combine_net_q = combine_net()
        # combine_net_k = combine_net()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        #for param_q, param_k in zip(
        #    self.diff_net_q.parameters(), self.diff_net_k.parameters()
        #):
        #    param_k.data.copy_(param_q.data)  # initialize
        #    param_k.requires_grad = False  # not update by gradient
        #
        #for param_q, param_k in zip(
        #    self.combine_net_q.parameters(), self.combine_net_k.parameters()
        #):
        #    param_k.data.copy_(param_q.data)  # initialize
        #    param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_img", torch.randn(K, dim))
        self.queue_img = nn.functional.normalize(self.queue_img, dim=1)

        self.register_buffer("queue_txt", torch.randn(K, dim))
        self.queue_txt = nn.functional.normalize(self.queue_txt, dim=1)

        #self.queue_cap = ["" for i in range(K)]

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, imgs, txts):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)
        #txts = concat_all_gather(keys)

        batch_size = imgs.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_img[ptr : ptr + batch_size, :] = imgs
        self.queue_txt[ptr : ptr + batch_size, :] = txts
        #self.queue_cap[ptr : ptr + batch_size] = caps
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # get batch size
        batch_size = x.shape[0]
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        # shuffle index for the gpu
        return x[idx_shuffle], idx_unshuffle

        
        ## gather from all gpus
        #batch_size_this = x.shape[0]
        #x_gather = concat_all_gather(x)
        #batch_size_all = x_gather.shape[0]
        #
        #num_gpus = batch_size_all // batch_size_this
        #
        ## random shuffle index
        #idx_shuffle = torch.randperm(batch_size_all).cuda()
        #
        ## broadcast to all gpus
        #torch.distributed.broadcast(idx_shuffle, src=0)
        #
        ## index for restoring
        #idx_unshuffle = torch.argsort(idx_shuffle)
        #
        ## shuffled index for this gpu
        #gpu_idx = torch.distributed.get_rank()
        #idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        #
        #return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        return x[idx_unshuffle]

        ## gather from all gpus
        #batch_size_this = x.shape[0]
        #x_gather = concat_all_gather(x)
        #batch_size_all = x_gather.shape[0]
        #
        #num_gpus = batch_size_all // batch_size_this
        #
        ## restored index for this gpu
        #gpu_idx = torch.distributed.get_rank()
        #idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        #
        #return x_gather[idx_this]

    def forward(self, img, txt, clip_model):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        #clip_model.eval()
        #for p in clip_model.named_parameters():
        #    print(p[0], p[1].requires_grad)
        # compute query features
        with torch.no_grad():
            img_features = clip_model.encode_image(img) # n*k
            txt_features = clip_model.encode_text(txt) # n*k
        img_features = nn.functional.normalize(img_features, dim=-1)
        txt_features = nn.functional.normalize(txt_features, dim=-1)

        # begin to retrieve for the similar samples
        retrieval_mat = torch.einsum("nc,kc->nk", [img_features, self.queue_txt.clone().detach()]) # n*k
        #retrieval_mat = torch.einsum("nc,kc->nk", [img_features, self.queue_img.clone().detach()])
        similarity_weight, ind_similar = retrieval_mat.max(dim=-1) # n*1, n*1
        t_similar_features = self.queue_txt.clone().detach()[ind_similar] # n*c
        #i_similar_features = self.queue_img.clone().detach()[ind_similar] # n*c
        #print(ind_similar)
        #cap_similar = self.queue_cap[ind_similar] # n
        #cap_similar = [self.queue_cap[i] for i in ind_similar]
        #prompts = ["Text1: %s. Text2: %s" % (cap[i], cap_similar[i]) for i in range(len(img_features))]
        
        #with concurrent.futures.ThreadPoolExecutor() as executor:
        #    diff_caps = list(executor.map(get_response, prompts))
        #print(diff_caps)
        #diff_caps = tokenizer(diff_caps).cuda()
        #with torch.no_grad():
        #    diff_features = clip_model.encode_text(diff_caps)
        #diff_features = nn.functional.normalize(diff_features, dim=-1)

        diff_features = self.diff_net_q(txt_features, t_similar_features) # normalized, n*c
        #diff_features_reverse = self.diff_net_q(t_similar_features, txt_features)
        combined_features = self.combine_net_q(img_features, diff_features) # normailzed, n*c
        #combined_features_reverse = self.combine_net_q(i_similar_features, diff_features_reverse)
        
        logit_scale = clip_model.logit_scale.exp()
        logit_scale = logit_scale.mean()

        l_retrieval = torch.einsum("nc,kc->nk",[combined_features, self.queue_img.clone().detach()])
        #l_retrieval_reverse = combined_features_reverse @ img_features.T
        logits = logit_scale * l_retrieval
        #logits2 = logit_scale * l_retrieval_reverse
        #gt2 = torch.arange(len(img_features)).long().cuda()

        self._dequeue_and_enqueue(img_features, txt_features)


        return logits, ind_similar
        



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output