#!/usr/bin/env python3
"""
build_lexicon_from_indiccorp.py
Builds a word-frequency lexicon file from AI4Bharat's IndicCorpV2
(https://huggingface.co/datasets/ai4bharat/IndicCorpV2) for any of this
project's 10 languages -- a much larger, general-domain source than the
narrow ~4,580-word lexicon built from just this project's own train/val
splits.

IMPORTANT CAVEAT (read before trusting results from this):
IndicCorpV2 is general-domain NEWS/WEB text, not signage-specific
vocabulary. It will add real coverage for genuine words currently missing
from the narrow training-set lexicon, but it will also add many words
that would never plausibly appear on a physical roadside sign, while
still lacking specialized business names and English loanwords that
actually do appear there. A bigger lexicon is not automatically better --
this project already found one reasonable-sounding refinement (frequency-
based tie-breaking) that turned out to be a net regression when tested.
Always compare Fixed-vs-Broken against the existing lexicon before
adopting this, exactly like every other optimization in this project.

Streams the dataset (does not download the full multi-GB/many-GB split
before processing) and stops after --max_rows sentences, since a full
Telugu split alone is ~31M rows -- far more than needed to build a large
vocabulary sample.

Usage:
    pip install datasets
    python build_lexicon_from_indiccorp.py --language telugu \
        --max_rows 500000 --min_frequency 2 \
        --out data/telugu/indiccorp_lexicon.tsv

Then feed the output into lexicon_rerank_eval.py or
lexicon_correct_predictions.py's --lexicon_ann alongside (not instead of)
the existing train/val TSVs:
    --lexicon_ann data/telugu/train/train.tsv data/telugu/val/val.tsv \\
                  data/telugu/indiccorp_lexicon.tsv
"""
import argparse
import os
import re
import sys
from collections import Counter

# Split names confirmed directly from the IndicCorpV2 dataset page
# (https://huggingface.co/datasets/ai4bharat/IndicCorpV2) -- matches this
# project's vocab.py UNICODE_BLOCKS naming convention on the left.
LANGUAGE_TO_SPLIT = {
    "telugu":    "tel_Telu",
    "tamil":     "tam_Taml",
    "malayalam": "mal_Mlym",
    "kannada":   "kan_Knda",
    "hindi":     "hin_Deva",
    "marathi":   "mar_Deva",
    "bengali":   "ben_Beng",
    "gujarati":  "guj_Gujr",
    "punjabi":   "pan_Guru",
    "odia":      "ory_Orya",
}

# Matches punctuation/digits this project's own vocab.py already treats as
# common extras (see _COMMON_EXTRAS) -- stripped from word boundaries so
# "word." and "word" count as the same token, without altering the word
# itself if punctuation appears mid-token (rare, left as-is rather than
# guessing how to split it).
_STRIP_CHARS = " \t\n0123456789.,:;!?()-'\"“”‘’—।"


def tokenize(text):
    return [w.strip(_STRIP_CHARS) for w in text.split()]


def stream_and_count(dataset_iterable, max_rows, min_word_len=1):
    """dataset_iterable: any iterable of dicts with a 'text' key (matches
    IndicCorpV2's schema) or plain strings -- kept generic so this can be
    tested with a simple mock list, not just a real HuggingFace dataset."""
    word_freq = Counter()
    n_rows = 0
    for row in dataset_iterable:
        if n_rows >= max_rows:
            break
        text = row["text"] if isinstance(row, dict) else row
        for w in tokenize(text):
            if w and len(w) >= min_word_len:
                word_freq[w] += 1
        n_rows += 1
    return word_freq, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=sorted(LANGUAGE_TO_SPLIT),
                     help="Any of this project's 10 supported languages")
    ap.add_argument("--max_rows", type=int, default=500_000,
                     help="Stop after this many sentences/rows (default 500k -- a full "
                          "language split can be tens of millions of rows; streaming stops "
                          "early rather than downloading/processing the entire split)")
    ap.add_argument("--min_frequency", type=int, default=2,
                     help="Drop words appearing fewer than this many times (default 2) -- "
                          "guards against including one-off tokenization noise/garbage as "
                          "if it were a real word")
    ap.add_argument("--min_word_len", type=int, default=1)
    ap.add_argument("--out", required=True,
                     help="Output TSV path, in the same "
                          "placeholder-image-path<TAB>word format lexicon_rerank_eval.py "
                          "and lexicon_correct_predictions.py already expect for --lexicon_ann")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("!! The 'datasets' package isn't installed. Run:")
        print("!!     pip install datasets")
        print("!! then re-run this script.")
        sys.exit(1)

    split_name = LANGUAGE_TO_SPLIT[args.language]
    print(f"Streaming AI4Bharat/IndicCorpV2, split '{split_name}' ({args.language})...")
    print(f"(streaming mode -- will stop after {args.max_rows:,} rows, not download the full split)")

    dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2",
                            split=split_name, streaming=True)

    word_freq, n_rows = stream_and_count(dataset, args.max_rows, args.min_word_len)
    print(f"Processed {n_rows:,} rows -> {len(word_freq):,} unique word type(s) before frequency filtering")

    filtered = {w: c for w, c in word_freq.items() if c >= args.min_frequency}
    print(f"After --min_frequency={args.min_frequency} filter: {len(filtered):,} word(s) kept "
          f"({len(word_freq) - len(filtered):,} dropped as likely noise)")

    if not filtered:
        print("!! No words survived filtering -- try a lower --min_frequency or more --max_rows.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (word, count) in enumerate(sorted(filtered.items(), key=lambda x: -x[1])):
            f.write(f"indiccorp_{i:07d}.png\t{word}\n")

    print(f"\nWrote {len(filtered):,} words to: {args.out}")
    print(f"\nReminder: this is general-domain text, not signage-specific. Compare")
    print(f"Fixed-vs-Broken against your existing lexicon before adopting it --")
    print(f"a bigger lexicon is not automatically a better one.")


if __name__ == "__main__":
    main()
