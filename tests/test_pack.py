"""Packed token stream: packer round-trip + memmap dataset windowing."""
import gc
import tempfile
from pathlib import Path

import numpy as np

from src.tokenizer.train_bpe import train_bpe
from src.data.pack_tokens import pack_corpus, load_meta
from src.train.train_clm import PackedCLMDataset


def _write_bin(path, ids, dtype=np.uint16):
    np.asarray(ids, dtype=dtype).tofile(path)


def _close_memmap(arr):
    """Release a numpy memmap so Windows can delete the backing file."""
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        mm.close()


def test_pack_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("teaching is hard. " * 200, encoding="utf-8")
        tok = train_bpe([corpus], tmp / "t.json",
                        vocab_size=400, min_frequency=1)

        out = tmp / "packed.bin"
        meta = pack_corpus(corpus, tok, out, dtype="uint16")

        assert meta["n_tokens"] > 0
        assert meta["dtype"] == "uint16"
        # file size matches token count * 2 bytes
        assert out.stat().st_size == meta["n_tokens"] * 2
        # sidecar written + loadable
        assert load_meta(out)["n_tokens"] == meta["n_tokens"]

        # content matches re-encoding the stripped line (packer strips lines)
        ids = tok.encode(("teaching is hard. " * 200).strip(),
                         add_special_tokens=False).ids
        data = np.memmap(out, dtype=np.uint16, mode="r")
        try:
            assert data.tolist() == ids
        finally:
            _close_memmap(data)
            del data
            gc.collect()


def test_pack_uint32():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("teaching is hard. " * 50, encoding="utf-8")
        tok = train_bpe([corpus], tmp / "t.json",
                        vocab_size=400, min_frequency=1)
        out = tmp / "p.bin"
        meta = pack_corpus(corpus, tok, out, dtype="uint32")
        assert meta["itemsize"] == 4
        assert out.stat().st_size == meta["n_tokens"] * 4


def test_packed_dataset_shapes_and_specials():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bin_path = tmp / "d.bin"
        # 100 tokens valued 10..109 (distinct from specials below)
        _write_bin(bin_path, list(range(10, 110)))

        cls_id, sep_id, pad_id = 1, 2, 0
        max_len = 16                       # body_len = 14
        ds = PackedCLMDataset(bin_path, cls_id=cls_id, sep_id=sep_id,
                              pad_id=pad_id, max_len=max_len, step=14,
                              dtype="uint16")
        try:
            # 100 tokens, step 14, min body 8 -> starts 0..92 -> 7 windows
            assert len(ds) == 7

            item = ds[0]
            assert item["input_ids"].shape[0] == max_len - 1
            assert item["labels"].shape[0] == max_len - 1
            # first token is [CLS]; body follows from the stream
            assert item["input_ids"][0].item() == cls_id
            assert item["input_ids"][1].item() == 10
            # window 0 = [CLS,10..23,SEP]; labels shift left so SEP is last
            assert item["labels"][-1].item() == sep_id
        finally:
            ds.close()
            gc.collect()


def test_packed_dataset_padding_labels():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bin_path = tmp / "d.bin"
        # exactly 10 tokens -> single short window, must pad
        _write_bin(bin_path, list(range(10, 20)))

        ds = PackedCLMDataset(bin_path, cls_id=1, sep_id=2, pad_id=0,
                              max_len=32, step=30, dtype="uint16")
        try:
            assert len(ds) == 1
            item = ds[0]
            # padded region in labels must be ignored (-100)
            assert (item["labels"] == -100).any()
            # no pad id (0) leaks as a real label
            assert (item["labels"] == 0).sum() == 0
        finally:
            ds.close()
            gc.collect()


def test_packed_dataset_empty():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bin_path = tmp / "d.bin"
        _write_bin(bin_path, list(range(10, 15)))  # 5 tokens < _MIN_BODY
        ds = PackedCLMDataset(bin_path, cls_id=1, sep_id=2, pad_id=0,
                              max_len=32, step=30, dtype="uint16")
        try:
            assert len(ds) == 0
        finally:
            ds.close()
            gc.collect()
