"""Task heads: MLM (pretrain) and QA span (fine-tune)."""
import torch
import torch.nn as nn


class CausalLMHead(nn.Module):
    """Predict next token vocab logits at each position. Optionally tie weight to embedding."""
    def __init__(self, d_model, vocab_size, tied_weight=None):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        if tied_weight is not None:
            self.decoder.weight = tied_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden):
        h = self.ln(self.act(self.transform(hidden)))
        return self.decoder(h) + self.bias


class QASpanHead(nn.Module):
    """Extractive QA: per-token start/end logits over the sequence."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 2)

    def forward(self, hidden, attention_mask=None):
        logits = self.proj(hidden)
        start, end = logits.split(1, dim=-1)
        start = start.squeeze(-1)
        end = end.squeeze(-1)
        if attention_mask is not None:
            mask = (attention_mask == 0)
            start = start.masked_fill(mask, -1e9)
            end = end.masked_fill(mask, -1e9)
        return start, end
