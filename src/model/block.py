"""Transformer-style decoder block with swappable attention."""
import torch.nn as nn

from .attention.softmax import SoftmaxAttention
from .attention.linear import LinearAttention
from .attention.rwkv import RWKVTimeMix


ATTENTION_REGISTRY = {
    'softmax': SoftmaxAttention,
    'linear': LinearAttention,
    'rwkv': RWKVTimeMix,
}


def build_attention(name, d_model, n_heads, dropout=0.0):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"unknown attention '{name}'. options: {list(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[name](d_model, n_heads, dropout)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)); x + FFN(LN(x))."""
    def __init__(self, d_model, n_heads, d_ff, attention='softmax', dropout=0.0):
        super().__init__()
        self.attn = build_attention(attention, d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x
