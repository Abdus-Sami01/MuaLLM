"""Decoder stack with embedding + N blocks."""
import torch
import torch.nn as nn

from .block import DecoderBlock


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input_ids):
        B, N = input_ids.shape
        if N > self.max_len:
            raise ValueError(f"sequence length {N} exceeds max_len {self.max_len}")
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok(input_ids) + self.pos(positions)
        return self.dropout(self.ln(x))


class Decoder(nn.Module):
    """Decoder-only LM body. Outputs hidden states (B, N, d_model)."""
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, max_len=512, attention='softmax',
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_len, dropout, pad_id)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, attention, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.pad_id = pad_id
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.attention_name = attention

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        return self.ln_f(x)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
