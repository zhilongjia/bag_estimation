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

        # net params for clinical data encoder
        self.input_dim_encoder = input_dim_encoder
        self.hidden_dim_encoder = args.hidden_dim_encoder
        self.encoder_layers = args.encoder_layers

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

        self.encoder = MLP(input_dim=self.input_dim_encoder, hidden_dim=args.hidden_dim,
                           out_dim=self.dim, num_layers=self.encoder_layers, res_conn=False)

        self.decoder = MLP(input_dim=self.dim, hidden_dim=args.hidden_dim, out_dim=self.input_dim_encoder,
                           num_layers=self.encoder_layers, res_conn=False)

        self.local_attn = LocalAttention(self.dim, self.attn_layers, self.n_heads, self.dim_head, self.dropout)

        self.predictor = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def get_attn_mask(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        attn_mask = torch.eye(n).to(self.device)
        attn_mask[:, 0] = 1.
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).repeat(1, self.n_heads, 1, 1).eq(0)
        return attn_mask

    def f_pooling(self, z):
        # z -> [b n d]
        if self.pooling == 'mean':
            out = torch.mean(z, dim=1)
        elif self.pooling == 'sum':
            out = torch.sum(z, dim=1)
        else:
            out = torch.sum(z, dim=1)
        return out

    def multimodal_data(self, img, x):
        h_img = self.vit(img)
        h_x = self.encoder(x)
        # local attention
        z = torch.cat([h_x.unsqueeze(1), h_img], dim=1)
        attn_mask = self.get_attn_mask(z)
        z = self.local_attn(z, attn_mask)
        z_img = z[:, 1:]
        z_pooled = self.f_pooling(z_img)
        out = self.predictor(z_pooled)
        return out

    def forward(self, img, mask=None):
        data = []
        for i in range(img.size(0)):
            x = img[i]
            x = self.transform(x)
            data.append(x)
        data = torch.stack(data, dim=0)
        if mask is not None:
            data = data * mask
        return self.vitonly(data)

    def pretrain(self, img, x):
        h_img = self.vit(img)
        h_x = self.encoder(x)
        # local attention
        z = torch.cat([h_x.unsqueeze(1), h_img], dim=1)
        attn_mask = self.get_attn_mask(z)
        z = self.local_attn(z, attn_mask)
        z_img = z[:, 1:]
        z_pooled = self.f_pooling(z_img)

        x_recon = self.decoder(h_x)

        return z_pooled, h_x, x_recon

    def vitonly(self, img):
        z = self.vit(img)
        z_pooled = self.f_pooling(z)
        out = self.predictor(z_pooled)
        return out
