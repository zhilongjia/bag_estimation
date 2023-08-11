import torch

import torch.nn as nn
from torch import Tensor
from monai import transforms as mo_trans
from model.base_model import LocalAttention, MLP
from model.transformer3D import ViT3D
from utils.datasets import get_transform


class BrainAgeEstimator(nn.Module):
    def __init__(self, args, device, channels, image_size, image_patch_size,
                 depth, depth_patch_size, input_dim_encoder):
        super().__init__()
        # net params for image
        self.image_size = image_size
        self.image_patch_size = image_patch_size
        self.depth = depth
        self.depth_patch_size = depth_patch_size
        self.channels = channels

        # dims for transformer/Attention
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim
        self.dim_mlp = args.dim_mlp
        self.n_heads = args.n_heads
        self.dim_head = args.dim_head
        self.attn_layers = args.attn_layers

        # dropout
        self.dropout = args.dropout
        self.emb_dropout = args.emb_dropout

        # Pooling
        self.pooling = args.pooling

        #
        self.transform = mo_trans.Compose([
            mo_trans.CenterSpatialCrop(roi_size=(182, 182, 182)),
            mo_trans.Resize(spatial_size=(128, 128, 128)),
            mo_trans.NormalizeIntensity(nonzero=True),
            mo_trans.ToTensor()
        ])

        self.device = device
        # modules
        self.vit = ViT3D(image_size, image_patch_size, depth, depth_patch_size,
                         self.dim, channels=self.channels, dim_mlp=self.dim_mlp,
                         dim_head=self.dim_head, attn_layers=self.attn_layers, n_heads=self.n_heads,
                         dropout=self.dropout, emb_dropout=self.emb_dropout)

        self.predictor = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, img):
        z = self.vit(img)
        z_pooled = self.f_pooling(z)
        out = self.predictor(z_pooled)
        return out

    def forward_with_mask(self, img, mask=None):
        data = []
        for i in range(img.size(0)):
            x = img[i]
            x = self.transform(x)
            data.append(x)
        data = torch.stack(data, dim=0)
        if mask is not None:
            data = data * mask
        return self.forward(data)
