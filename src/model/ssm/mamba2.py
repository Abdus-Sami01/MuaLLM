"""Mamba2 SSM block as a drop-in 'attention' for the registry.

Native Mamba2 (mamba-ssm pip package) needs CUDA + Triton. CPU smoke-test path
falls back to a tiny pure-PyTorch diagonal state-space (S4D-lite) so unit tests
keep running on CPU without the heavy dependency.

Both expose the same signature as the existing attention variants:
    __init__(d_model, n_heads, dropout=0.0)
    forward(x, attn_mask=None) -> x   # x: (B, N, D)

`n_heads` is unused (Mamba is not head-based) but kept for API parity so the
registry can construct it with the same call.

Config key: `attention: mamba`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba2 as _Mamba2Native  # type: ignore
    HAS_NATIVE_MAMBA = True
except (ImportError, OSError, RuntimeError):
    _Mamba2Native = None
    HAS_NATIVE_MAMBA = False


class _S4DLite(nn.Module):
    """Tiny diagonal SSM for CPU fallback.

    Per-channel diagonal A in (-inf, 0) via -exp(A_log). Discretized with
    learnable per-channel timestep `dt`:

        dA = exp(dt * A)             (D, S)
        dB = dt * B                  (D, S)
        x_t = dA * x_{t-1} + dB * u_t
        y_t = sum_s(C * x_t) + D * u_t

    No selective gating, no parallel scan - this is a placeholder so CPU smoke
    tests pass. For real training use the GPU native path.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.zeros(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        # softplus(-2) ~= 0.127  -> sensible default timestep
        self.dt_raw = nn.Parameter(torch.full((d_model,), -2.0))

    def forward(self, u):
        # u: (B, L, D)
        B, L, D = u.shape
        dt = F.softplus(self.dt_raw)                  # (D,)
        A = -torch.exp(self.A_log)                    # (D, S)
        dA = torch.exp(dt[:, None] * A)               # (D, S)
        dB = dt[:, None] * self.B                     # (D, S)

        x = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            u_t = u[:, t]                             # (B, D)
            x = dA[None] * x + dB[None] * u_t[:, :, None]
            y_t = (self.C[None] * x).sum(dim=-1) + self.D * u_t
            ys.append(y_t)
        return torch.stack(ys, dim=1)                 # (B, L, D)


class MambaMix(nn.Module):
    """Registry-compatible Mamba block.

    On CUDA with mamba-ssm available, wraps `Mamba2(d_model, d_state, d_conv, expand)`.
    Otherwise uses `_S4DLite` so CPU paths still work.
    """
    def __init__(self, d_model, n_heads, dropout=0.0,
                 d_state=64, d_conv=4, expand=2):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads  # unused, parity only

        if HAS_NATIVE_MAMBA and torch.cuda.is_available():
            self.core = _Mamba2Native(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
            self.is_native = True
        else:
            self.core = _S4DLite(d_model, d_state=min(d_state, 16))
            self.is_native = False

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, N, D)
        if attn_mask is not None:
            m = attn_mask[..., None].to(x.dtype)
            x = x * m
        y = self.core(x)
        if attn_mask is not None:
            y = y * attn_mask[..., None].to(y.dtype)
        y = self.dropout(y)
        return self.out_proj(y)
