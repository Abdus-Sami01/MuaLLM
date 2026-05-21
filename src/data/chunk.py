"""Chunk text into token-length windows with overlap."""


def chunk_text(text, tokenizer, max_tokens=256, stride=64):
    """
    Tokenize without special tokens, slide window over ids.
    Returns list[list[int]] of chunks (no [CLS]/[SEP] added here).
    """
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = enc.ids
    if not ids:
        return []
    chunks = []
    step = max(1, max_tokens - stride)
    i = 0
    while i < len(ids):
        chunk = ids[i:i + max_tokens]
        if len(chunk) < 8:
            break
        chunks.append(chunk)
        if i + max_tokens >= len(ids):
            break
        i += step
    return chunks


def chunk_files(text_files, tokenizer, max_tokens=256, stride=64):
    """Chunk many files. Returns flat list of chunks."""
    all_chunks = []
    for p in text_files:
        text = open(p, "r", encoding="utf-8", errors="ignore").read()
        all_chunks.extend(chunk_text(text, tokenizer, max_tokens, stride))
    return all_chunks
