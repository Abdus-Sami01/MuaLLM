"""Encoder + heads integration tests."""
import pytest
import torch

from src.model.encoder import Encoder
from src.model.heads import MLMHead, QASpanHead


VARIANTS = ["softmax", "linear", "rwkv"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_encoder_forward(variant):
    torch.manual_seed(0)
    vocab = 100
    enc = Encoder(vocab_size=vocab, d_model=32, n_heads=2, n_layers=2,
                  d_ff=64, max_len=32, attention=variant, pad_id=0)
    ids = torch.randint(1, vocab, (2, 16))
    hidden = enc(ids)
    assert hidden.shape == (2, 16, 32)
    assert torch.isfinite(hidden).all()


def test_mlm_head_tied():
    enc = Encoder(vocab_size=50, d_model=16, n_heads=2, n_layers=1,
                  d_ff=32, max_len=16)
    mlm = MLMHead(16, 50, tied_weight=enc.embed.tok.weight)
    ids = torch.randint(1, 50, (1, 8))
    hidden = enc(ids)
    logits = mlm(hidden)
    assert logits.shape == (1, 8, 50)
    # weight tied to embedding
    assert mlm.decoder.weight.data_ptr() == enc.embed.tok.weight.data_ptr()


def test_qa_head_mask():
    head = QASpanHead(16)
    hidden = torch.randn(1, 10, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    start, end = head(hidden, attention_mask=mask)
    # padded positions should be -inf-ish
    assert (start[0, 5:] < -1e8).all()
    assert (end[0, 5:] < -1e8).all()
