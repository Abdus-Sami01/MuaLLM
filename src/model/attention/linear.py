"""Linear attention. Katharopoulos et al., 2020. O(N D^2). Bidirectional."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x):
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """
    softmax(Q K^T) V approximated by phi(Q) (phi(K)^T V) / (phi(Q) sum_n phi(K_n))
    with phi(x) = elu(x) + 1 ensuring non-negative kernel.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        if attn_mask is not None:
            m = attn_mask[:, None, :, None].to(q.dtype)
            k = k * m
            v = v * m

        # causal formulation
        k_cumsum = torch.cumsum(k, dim=2)
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        
        num = torch.einsum('bhnd,bhnde->bhne', q, kv_cumsum)
        denom = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum).clamp(min=self.eps)
        out = num / denom.unsqueeze(-1)

        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
