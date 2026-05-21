"""RWKV-style time-mix block. Bidirectional encoder variant. O(N D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wkv_forward(k, v, w_log_decay, u_bonus):
    """
    Numerically-stable causal WKV recurrence.
    k, v: (B, H, N, Dh)
    w_log_decay: (H, Dh)  -- log of decay in (-inf, 0)
    u_bonus:     (H, Dh)  -- current-step bonus
    Returns: (B, H, N, Dh)
    """
    B, H, N, D = k.shape
    out = torch.zeros_like(v)
    NEG_INF = torch.full((B, H, D), -1e30, device=k.device, dtype=k.dtype)

    num = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    den = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    max_log = NEG_INF.clone()

    w_log = w_log_decay  # (H, D)

    for t in range(N):
        kt = k[:, :, t]
        vt = v[:, :, t]

        kt_b = kt + u_bonus
        new_max = torch.maximum(max_log + w_log, kt_b)
        e1 = torch.exp(max_log + w_log - new_max)
        e2 = torch.exp(kt_b - new_max)
        out_num = e1 * num + e2 * vt
        out_den = e1 * den + e2
        out[:, :, t] = out_num / (out_den + 1e-6)

        new_max_state = torch.maximum(max_log + w_log, kt)
        es1 = torch.exp(max_log + w_log - new_max_state)
        es2 = torch.exp(kt - new_max_state)
        num = es1 * num + es2 * vt
        den = es1 * den + es2
        max_log = new_max_state

    return out


class RWKVTimeMix(nn.Module):
    """
    Bidirectional RWKV-style time-mix for encoder QA.
    Forward direction + reverse direction, concatenated, projected back.
    Pure PyTorch (Python loop) - slow for long sequences but CPU-friendly.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, bidirectional=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bidirectional = bidirectional

        self.mix_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mix_r = nn.Parameter(torch.full((1, 1, d_model), 0.5))

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

        # parameterize decay as -exp(w_raw) so it lies in (-inf, 0)
        self.time_decay_raw = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.time_bonus = nn.Parameter(torch.zeros(n_heads, self.d_head))

        out_dim = 2 * d_model if bidirectional else d_model
        self.out_proj = nn.Linear(out_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _shift_right(x):
        return F.pad(x, (0, 0, 1, 0))[:, :-1]

    @staticmethod
    def _shift_left(x):
        return F.pad(x, (0, 0, 0, 1))[:, 1:]

    def _project(self, x_orig, shifted):
        xk = x_orig * self.mix_k + shifted * (1 - self.mix_k)
        xv = x_orig * self.mix_v + shifted * (1 - self.mix_v)
        xr = x_orig * self.mix_r + shifted * (1 - self.mix_r)
        B, N, _ = x_orig.shape
        k = self.key(xk).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.value(xv).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        r = self.receptance(xr).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        return k, v, r

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        w_log = -torch.exp(self.time_decay_raw)  # (H, Dh), negative

        # forward direction
        k_f, v_f, r_f = self._project(x, self._shift_right(x))
        if attn_mask is not None:
            m = attn_mask[:, None, :, None].to(k_f.dtype)
            k_f = k_f.masked_fill(m == 0, -1e9)
            v_f = v_f * m
        wkv_f = _wkv_forward(k_f, v_f, w_log, self.time_bonus)
        out_f = torch.sigmoid(r_f) * wkv_f

        if not self.bidirectional:
            out = out_f.transpose(1, 2).contiguous().view(B, N, D)
            return self.out_proj(self.dropout(out))

        # reverse direction: flip sequence, run causal, flip back
        x_rev = x.flip(dims=[1])
        k_b, v_b, r_b = self._project(x_rev, self._shift_right(x_rev))
        if attn_mask is not None:
            m_rev = attn_mask.flip(dims=[1])[:, None, :, None].to(k_b.dtype)
            k_b = k_b.masked_fill(m_rev == 0, -1e9)
            v_b = v_b * m_rev
        wkv_b = _wkv_forward(k_b, v_b, w_log, self.time_bonus)
        out_b = torch.sigmoid(r_b) * wkv_b
        # un-flip back to original token order
        out_b = out_b.flip(dims=[2])

        out = torch.cat([out_f, out_b], dim=-1)  # (B, H, N, 2*Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, 2 * D)
        return self.out_proj(self.dropout(out))
