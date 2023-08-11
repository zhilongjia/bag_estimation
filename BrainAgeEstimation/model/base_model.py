import torch
import torch.nn as nn
from einops import repeat, rearrange


NEG_INF = -1e-9


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class LocalAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads=6, dim_head=64, dropout=0.):
        super(LocalAttentionLayer, self).__init__()

        h_dim = dim_head * n_heads
        self.dim = dim
        project_out = not (n_heads == 1 and dim_head == dim)
        self.n_heads = n_heads

        self.to_qkv = nn.Linear(dim, h_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.scale = dim_head ** -0.5

    def forward(self, x, attn_mask):
        b, n, d = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        # [b h n d] [b h d n] -> [b h n n]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores.masked_fill_(attn_mask, NEG_INF)
        attn = self.attend(scores)
        context = torch.matmul(attn, v)

        context = rearrange(context, 'b h n d -> b n (h d)')
        out = self.out_layer(context) + x
        return out


class LocalAttention(nn.Module):
    def __init__(self, dim, attn_layers=1, n_heads=6, dim_head=64, dropout=0.):
        super().__init__()

        self.local_attn = PreNorm(dim, LocalAttentionLayer(dim, dim_head=dim_head, n_heads=n_heads, dropout=dropout))

    def forward(self, x, attn_mask):
        z = self.local_attn(x, attn_mask)
        return z


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=2, activation=nn.ReLU, res_conn=True):
        super(MLP, self).__init__()
        self.res_conn = res_conn
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

        self.out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.net(x) + self.shortcut(x) if self.res_conn else self.net(x)
        out = self.out(h)
        return out



