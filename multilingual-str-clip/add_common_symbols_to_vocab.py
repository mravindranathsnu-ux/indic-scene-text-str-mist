#!/usr/bin/env python3
"""
add_common_symbols_to_vocab.py
Proactively adds a curated set of common special symbols to every
language's vocab file -- digits, punctuation, currency signs, and basic
Latin letters -- so real-world signage containing prices, phone numbers,
English brand names, or common punctuation doesn't hit unrecoverable
<unk> tokens at inference time.

This is DIFFERENT from check_all_vocab_coverage.py, which is reactive
(only catches characters already present in your current training data).
This is proactive: it adds symbols that COMMONLY appear on real Indian
signage even if your current dataset happens not to contain them yet.

Like check_all_vocab_coverage.py, this only helps FUTURE retrains --
already-trained checkpoints embed their own vocab snapshot and are
unaffected (safe to run any time, including on languages you don't plan
to retrain right away).

Usage:
    python add_common_symbols_to_vocab.py                    # all languages, dry run
    python add_common_symbols_to_vocab.py --fix               # actually apply
    python add_common_symbols_to_vocab.py --languages telugu tamil --fix
"""
import argparse
import json
import os

# Curated, conservative set -- common on real signage, not an exhaustive
# "every possible symbol" dump. Grouped for readability; order doesn't matter.
DIGITS = list("0123456789")
BASIC_PUNCTUATION = list(".,:;!?-_()/\\&@#%*+=\"'` ")
CURRENCY = ["\u20b9", "$"]  # rupee, dollar (common on price signage)
LATIN_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LATIN_LOWER = list("abcdefghijklmnopqrstuvwxyz")

SYMBOL_GROUPS = {
    "digits": DIGITS,
    "punctuation": BASIC_PUNCTUATION,
    "currency": CURRENCY,
    "latin_upper": LATIN_UPPER,
    "latin_lower": LATIN_LOWER,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--languages", nargs="+", default=None,
                     help="Languages to update. Default: every *_vocab.json found in vocab/")
    ap.add_argument("--groups", nargs="+", default=list(SYMBOL_GROUPS.keys()),
                     choices=list(SYMBOL_GROUPS.keys()),
                     help="Which symbol groups to add (default: all groups)")
    ap.add_argument("--fix", action="store_true", help="Actually write changes; default is dry-run")
    args = ap.parse_args()

    vocab_dir = os.path.join(args.project_root, "vocab")
    if args.languages:
        langs = args.languages
    else:
        langs = sorted(
            f[:-len("_vocab.json")] for f in os.listdir(vocab_dir)
            if f.endswith("_vocab.json")
        )

    to_add_all = []
    for group in args.groups:
        to_add_all.extend(SYMBOL_GROUPS[group])
    # de-dup while preserving order
    seen = set()
    to_add_all = [c for c in to_add_all if not (c in seen or seen.add(c))]

    print(f"Symbol groups requested: {', '.join(args.groups)}")
    print(f"Total candidate symbols: {len(to_add_all)}")
    print()
    print(f"{'Language':<14s} {'Vocab (before)':>15s} {'Missing':>9s}  Adding")
    print("-" * 80)

    results = []
    for lang in langs:
        vocab_path = os.path.join(vocab_dir, f"{lang}_vocab.json")
        if not os.path.isfile(vocab_path):
            print(f"{lang:<14s}  (no vocab file found -- skipped)")
            continue
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        vocab_set = set(vocab)
        missing = [c for c in to_add_all if c not in vocab_set]
        results.append((lang, vocab_path, vocab, missing))
        missing_display = "".join(c if c != " " else "\u2423" for c in missing)  # visualize space
        if len(missing_display) > 40:
            missing_display = missing_display[:40] + f"...(+{len(missing)-40} more)"
        print(f"{lang:<14s} {len(vocab):>15d} {len(missing):>9d}  {missing_display or '-'}")

    if args.fix:
        print("\n--- Applying ---")
        for lang, vocab_path, vocab, missing in results:
            if not missing:
                continue
            new_vocab = vocab + missing
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(new_vocab, f, ensure_ascii=False, indent=2)
            print(f"{lang}: {len(vocab)} -> {len(new_vocab)} (+{len(missing)})")
        print("\nDone. This only affects FUTURE retrains of these languages.")
    else:
        any_missing = any(m for _, _, _, m in results)
        if any_missing:
            print("\nRun again with --fix to apply.")
        else:
            print("\nAll checked languages already have full coverage of the requested symbol groups.")


if __name__ == "__main__":
    main()
