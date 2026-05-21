"""Train a byte-level BPE tokenizer with special tokens for QA."""
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_bpe(text_files, output_path, vocab_size=8000, min_frequency=2):
    """Train BPE on a list of plain-text files. Save tokenizer.json."""
    text_files = [str(p) for p in text_files]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(text_files, trainer)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    return tokenizer


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))


def special_token_ids(tokenizer):
    return {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="paths to plain text files")
    ap.add_argument("--out", required=True, help="path to save tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()
    tok = train_bpe(args.inputs, args.out, args.vocab_size, args.min_frequency)
    print(f"vocab_size={tok.get_vocab_size()} -> {args.out}")
