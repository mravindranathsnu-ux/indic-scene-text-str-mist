#!/usr/bin/env python3
"""
check_all_vocab_coverage.py
Checks every language's vocab file against its actual train.tsv content,
reporting missing characters for all of them in one pass -- the same check
sanity_check.py does per-language, but run comprehensively across the
whole project so gaps don't only get caught one language at a time.

Usage:
    python check_all_vocab_coverage.py
    python check_all_vocab_coverage.py --fix     # also patches each vocab
                                                   # file, appending only
                                                   # the characters found
                                                   # missing (never removes
                                                   # or reorders anything)
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--fix", action="store_true",
                     help="Append missing characters directly to each vocab file")
    args = ap.parse_args()

    data_dir = os.path.join(args.project_root, "data")
    vocab_dir = os.path.join(args.project_root, "vocab")

    languages = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    print(f"{'Language':<16s} {'Vocab':>7s} {'Train chars':>12s} {'Missing':>9s}  Missing characters")
    print("-" * 90)

    results = []
    for lang in languages:
        train_tsv = os.path.join(data_dir, lang, "train", "train.tsv")
        vocab_path = os.path.join(vocab_dir, f"{lang}_vocab.json")

        if not os.path.isfile(train_tsv) or not os.path.isfile(vocab_path):
            continue  # language not set up yet -- nothing to check

        vocab = json.load(open(vocab_path, encoding="utf-8"))
        vocab_set = set(vocab)

        train_chars = set()
        with open(train_tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    train_chars.update(parts[1])

        missing = sorted(train_chars - vocab_set)
        results.append((lang, vocab_path, vocab, missing))

        missing_str = "".join(missing) if missing else "-"
        print(f"{lang:<16s} {len(vocab):>7d} {len(train_chars):>12d} {len(missing):>9d}  {missing_str}")

    if args.fix:
        print("\n--- Applying fixes ---")
        for lang, vocab_path, vocab, missing in results:
            if not missing:
                continue
            new_vocab = vocab + missing
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(new_vocab, f, ensure_ascii=False, indent=2)
            print(f"{lang}: added {len(missing)} character(s) -> vocab size {len(vocab)} -> {len(new_vocab)}")
        print("\nDone. Re-run without --fix to confirm every language now shows 0 missing.")
    else:
        any_missing = any(m for _, _, _, m in results)
        if any_missing:
            print("\nRun again with --fix to append the missing characters to each affected vocab file.")
        else:
            print("\nAll languages: full coverage, nothing to fix.")


if __name__ == "__main__":
    main()
