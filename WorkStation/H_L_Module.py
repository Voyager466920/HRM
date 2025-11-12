import math

import torch
import torch.nn as nn

class HModule(nn.Module):
    def __init__(self, num_heads:int=4, hidden_dim:int=256, ffn_mult:int=4, dropout:float=0.0 ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.qkv = nn.Linear(in_features= hidden_dim, out_features= hidden_dim * 3, bias=False)
        self.rope = RoPE()
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = RMSNorm(dim=hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * ffn_mult, bias=False),
            nn.SiLU(), #원래는 SwiGLU인데, Pytorch에 구현된게 없으므로 SiLU 사용.
            nn.Linear(in_features= hidden_dim * ffn_mult, out_features= hidden_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.norm2 = RMSNorm(dim=hidden_dim)


    def forward(self, x, z_H_previous, z_L_current, attn_mask=None):
        x = z_H_previous + z_L_current
        hidden = self.norm1(x)
        batch_size, sequence_length, hidden_size = hidden.shape()
        qkv = self.qkv(hidden).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q)
        k = self.rope(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        x = x + self.proj(out)
        h = self.norm2(x)
        x = x + self.ff(h)
        return x

import math
import torch
import torch.nn as nn

class LModule(nn.Module):
    def __init__(self, num_heads: int = 4, hidden_dim: int = 256, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.rope = RoPE(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, z_H_previous, z_L_current, attn_mask=None):
        x = z_H_previous + z_L_current
        h = self.norm1(x)
        b, l, d = h.shape
        qkv = self.qkv(h).view(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q)
        k = self.rope(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.out_proj(out)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# These improvements include Rotary Positional Encoding, Gated Linear Units, RMSNorm, and the removal of bias terms from linear layers.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, x):
        b, h, l, d = x.shape
        t = torch.arange(l, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,f->lf", t, self.inv_freq)
        cos, sin = freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rot

