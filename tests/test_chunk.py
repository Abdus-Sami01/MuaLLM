"""Chunking sanity."""
import tempfile
from pathlib import Path

from src.tokenizer.train_bpe import train_bpe
from src.data.chunk import chunk_text


def test_chunk_basic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("teaching is hard. " * 200, encoding="utf-8")
        tok = train_bpe([corpus], tmp / "t.json",
                        vocab_size=400, min_frequency=1)
        chunks = chunk_text("teaching is hard. " * 200, tok,
                            max_tokens=32, stride=8)
        assert len(chunks) > 1
        assert all(len(c) <= 32 for c in chunks)
        assert all(len(c) >= 8 for c in chunks)


def test_chunk_empty():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("teaching is hard.", encoding="utf-8")
        tok = train_bpe([corpus], tmp / "t.json",
                        vocab_size=400, min_frequency=1)
        chunks = chunk_text("", tok, max_tokens=32, stride=8)
        assert chunks == []
