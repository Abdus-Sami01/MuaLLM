"""Scaled dot-product multi-head attention. O(N^2 D). Baseline."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
        
        if attn_mask is not None:
            keep = attn_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~keep, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
