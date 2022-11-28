#!/usr/bin/env python3

"""
Vision Transformer (ViT) implementation.
"""

import os
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from iopath.common.file_io import PathManagerFactory

pathmgr = PathManagerFactory.get()




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer
        referene:
            - MAE:  https://github.com/facebookresearch/mae/blob/main/models_vit.py
            - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # remove the classifier
        if hasattr(self, "pre_logits"):
            del self.pre_logits
        del self.head

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 0].detach().float()
        return x
    
    def extract_feat_all(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = x.detach().float()
        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))
    
    def forward_bbox(self, x):
        all_tokens = self.forward_norm(self.extract_feat_all(x))
        # import ipdb; ipdb.set_trace()
        
        class_tokens = all_tokens[:, 0]
        patch_tokens = all_tokens[:, 1:]
        
        # reshape patch tokens to (B, H, W, C)
        patch_tokens_spatial = patch_tokens.reshape(x.shape[0], 14, 14, -1)
        patch_tokens_spatial = patch_tokens_spatial.permute(0, 3, 1, 2)
        return 0
        
        
        
        
    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        #print("Trainable parameters in the encoder:")
        #print(trainable_params)

def vit_b16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained.startswith("none")
    # load from checkpoint
    if not pretrained.startswith("none"):
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 768
    return model, hidden_dim

def vit_l16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained.startswith("none")
    # load from checkpoint
    if not pretrained.startswith("none"):
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 1024
    return model, hidden_dim


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    assert pathmgr.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    state_dict = checkpoint["model"]

    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")
        
        
        
        
        
        
        

import math
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from tqdm import tqdm

# Per-channel mean and standard deviation (in RGB order)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_im_size = 224
_max_shift = 0
_lower_crop = False

def lower_center_crop(im, crop_size):
    """Performs lower-center cropping."""
    w = im.shape[1]
    x = math.ceil((w - crop_size) / 2)
    return im[-crop_size:, x:(x + crop_size), :]


def lower_random_shift_crop(im, crop_size, max_shift):
    w = im.shape[1]
    x = (w - crop_size) // 2
    assert x + max_shift + crop_size <= w
    assert x - max_shift >= 0
    shift = np.random.randint(-max_shift, max_shift + 1)
    x = x + shift
    return im[-crop_size:, x:(x + crop_size), :]


def resize_crop(im, size, max_shift=0, lower_crop=True):
    """Performs image resize and crop."""
    if max_shift > 0:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_random_shift_crop(im, crop_size, max_shift)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_center_crop(im, crop_size)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    return im

def normalize(im):
    """Performs image normalization."""
    # [0, 255] -> [0, 1]
    im = im.astype(np.float32) / 255.0
    # Color norm
    im = color_norm(im, _MEAN, _STD)
    # HWC -> CHW
    im = im.transpose([2, 0, 1])
    return im


def color_norm(im, mean, std):
    """Performs per-channel normalization."""
    for i in range(3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im

def process_image(im):
    im_pil = Image.fromarray(im).convert("RGB")
    # im_pil = self._augmentation(im_pil)
    im = np.array(im_pil).astype(np.float32)
    im = resize_crop(im, _im_size, max_shift=_max_shift, lower_crop=_lower_crop)
    im = normalize(im)
    return im






# # vit_hands, _ = vit_l16("/private/home/jathushan/3D/mvp/mvp-l.pth")
# vit_hands, _ = vit_b16("/private/home/jathushan/3D/mvp/mvp-b.pth")

# img = cv2.imread("output.png")
# # crop by hand to get 480x480 bounding box path
# img = process_image(img)
# print(img.shape)
# img = torch.from_numpy(img).unsqueeze(0).float()
# print(img.shape)

# feat = vit_hands.forward_bbox(img)