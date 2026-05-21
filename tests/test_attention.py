"""Shape + mask sanity tests for the 3 attention variants."""
import pytest
import torch

from src.model.attention.softmax import SoftmaxAttention
from src.model.attention.linear import LinearAttention
from src.model.attention.rwkv import RWKVTimeMix


VARIANTS = [
    ("softmax", SoftmaxAttention),
    ("linear", LinearAttention),
    ("rwkv", RWKVTimeMix),
]


@pytest.mark.parametrize("name,cls", VARIANTS)
def test_forward_shape(name, cls):
    torch.manual_seed(0)
    B, N, D, H = 2, 16, 32, 4
    attn = cls(D, H)
    x = torch.randn(B, N, D)
    y = attn(x)
    assert y.shape == (B, N, D), f"{name}: got {y.shape}"


@pytest.mark.parametrize("name,cls", VARIANTS)
def test_mask_changes_output(name, cls):
    """With a non-trivial pad mask, output should differ from no-mask."""
    torch.manual_seed(0)
    B, N, D, H = 2, 16, 32, 4
    attn = cls(D, H)
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.long)
    mask[:, N // 2:] = 0
    y_masked = attn(x, attn_mask=mask)
    y_unmasked = attn(x)
    diff = (y_masked - y_unmasked).abs().mean().item()
    assert diff > 1e-6, f"{name}: mask had no effect (diff={diff})"


@pytest.mark.parametrize("name,cls", VARIANTS)
def test_backward(name, cls):
    """Gradients flow through to all parameters."""
    torch.manual_seed(0)
    B, N, D, H = 2, 8, 16, 2
    attn = cls(D, H)
    x = torch.randn(B, N, D, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    for p in attn.parameters():
        assert p.grad is not None, f"{name}: param has no grad"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad"


@pytest.mark.parametrize("name,cls", VARIANTS)
def test_no_nan_with_full_mask_row(name, cls):
    """Even with extreme mask (only first token kept), output is finite."""
    torch.manual_seed(0)
    B, N, D, H = 1, 8, 16, 2
    attn = cls(D, H)
    x = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.long)
    mask[:, 0] = 1
    y = attn(x, attn_mask=mask)
    assert torch.isfinite(y).all(), f"{name}: produced non-finite output"
