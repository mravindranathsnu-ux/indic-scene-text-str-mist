#!/usr/bin/env python3
"""
bstd_vocab_coverage.py

Checks whether BSTD's text (per language) is fully covered by the existing
Phase 1 (IIIT-IndicSTR-Word) trained vocabulary for that language. Any
uncovered character would silently become an <unk> token during evaluation,
deflating accuracy for reasons unrelated to real recognition performance.

Usage (run from the multilingual-str-clip project root, with bstd_processed/
available or pointed to via --bstd_dir):
    python3 bstd_vocab_coverage.py --bstd_dir /path/to/bstd_processed \
        --vocab_dir vocab
"""
import json
import argparse
import csv
from pathlib import Path
from collections import Counter

# BSTD language name -> Phase 1 vocab filename stem, where they differ
LANG_MAP = {
    'bengali': 'bengali', 'gujarati': 'gujarati', 'hindi': 'hindi',
    'kannada': 'kannada', 'malayalam': 'malayalam', 'marathi': 'marathi',
    'odia': 'odia', 'punjabi': 'punjabi', 'tamil': 'tamil', 'telugu': 'telugu',
    # english and assamese have no Phase 1 checkpoint -- flagged, not compared
}


def load_vocab_chars(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        data = json.load(f)
    # Vocab files may store a char->id dict, an id->char dict, or a list;
    # handle the common shapes defensively rather than assuming one.
    # Key signal: id->char dicts have keys that are digit strings ("0","1",...);
    # char->id dicts have keys that are the actual (non-digit) characters/tokens.
    if isinstance(data, dict):
        sample_keys = list(data.keys())[:50]
        keys_are_digit_ids = all(k.isdigit() for k in sample_keys)
        return set(data.values()) if keys_are_digit_ids else set(data.keys())
    elif isinstance(data, list):
        return set(data)
    else:
        raise ValueError(f"Unrecognized vocab file shape: {vocab_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bstd_dir', default='bstd_processed')
    ap.add_argument('--vocab_dir', default='vocab')
    args = ap.parse_args()

    bstd_dir = Path(args.bstd_dir)
    vocab_dir = Path(args.vocab_dir)

    print("=" * 70)
    print(f"{'Language':<12} {'VocabFile':<10} {'BSTD chars':>11} {'Uncovered':>10} {'Uncov %':>8}")
    print("=" * 70)

    results = {}
    for lang, vocab_stem in LANG_MAP.items():
        vocab_path = vocab_dir / f'{vocab_stem}_vocab.json'
        train_tsv = bstd_dir / lang / 'train' / 'train.tsv'
        test_tsv = bstd_dir / lang / 'test' / 'test.tsv'

        if not vocab_path.exists():
            print(f"{lang:<12} {'MISSING':<10} {'--':>11} {'--':>10} {'--':>8}   (no vocab file found at {vocab_path})")
            continue
        if not train_tsv.exists():
            print(f"{lang:<12} {'--':<10} {'MISSING':>11}   (no BSTD TSV found at {train_tsv})")
            continue

        vocab_chars = load_vocab_chars(vocab_path)

        bstd_chars = Counter()
        for tsv_path in [train_tsv, test_tsv]:
            if not tsv_path.exists():
                continue
            with open(tsv_path, encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    bstd_chars.update(row['ground_truth'])

        uncovered = {c: n for c, n in bstd_chars.items() if c not in vocab_chars}
        uncov_pct = 100 * sum(uncovered.values()) / sum(bstd_chars.values()) if bstd_chars else 0

        results[lang] = {'vocab_size': len(vocab_chars), 'bstd_chars': len(bstd_chars),
                          'uncovered': uncovered, 'uncov_pct': uncov_pct}

        flag = "  !!" if uncov_pct > 0.5 else ""
        print(f"{lang:<12} {'OK':<10} {len(bstd_chars):>11} {len(uncovered):>10} {uncov_pct:>7.2f}%{flag}")

    print()
    print("Languages with no Phase 1 checkpoint (skipped, need new training if used):")
    print("  english, assamese")

    print()
    print("=" * 70)
    print("Detail: top uncovered characters per language (if any)")
    print("=" * 70)
    for lang, r in results.items():
        if r['uncovered']:
            top = sorted(r['uncovered'].items(), key=lambda x: -x[1])[:15]
            print(f"\n  {lang}: {len(r['uncovered'])} uncovered character(s), most frequent first:")
            for ch, n in top:
                print(f"    {ch!r}  (U+{ord(ch):04X})  -- {n} occurrences")


if __name__ == '__main__':
    main()
