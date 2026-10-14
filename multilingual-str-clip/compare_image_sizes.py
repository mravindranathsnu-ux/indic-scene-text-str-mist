#!/usr/bin/env python3
"""
compare_image_sizes.py
Tests the "Gujarati's poor accuracy comes from tiny image crops" hypothesis
in two ways:

  1. ACROSS languages: average / median test-crop dimensions per language.
     If Gujarati's crops are markedly smaller than healthy languages', small
     inputs (heavily upscaled to the model's 224px) are a plausible root
     cause of its uniformly elevated error rate.

  2. WITHIN a language: crop width broken down by word-length bucket
     (the same 1-4 / 5-6 / 7-8 / 9-12 / 13+ buckets used in eval analysis).
     Gujarati's accuracy is INVERTED by length (short words worst) — if
     short words also have the smallest crops, the inversion tracks image
     size, not language statistics.

Usage (from the project root):
    python compare_image_sizes.py
    python compare_image_sizes.py --languages gujarati odia punjabi marathi
    python compare_image_sizes.py --sample 500 --detail gujarati odia
"""
import argparse
import os
import random
from PIL import Image


def read_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def bucket(ln):
    if ln <= 4: return "1-4"
    if ln <= 6: return "5-6"
    if ln <= 8: return "7-8"
    if ln <= 12: return "9-12"
    return "13+"


BUCKET_ORDER = ["1-4", "5-6", "7-8", "9-12", "13+"]


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def analyze_language(lang, project_root, sample_n, rng):
    tsv = os.path.join(project_root, "data", lang, "test", "test.tsv")
    img_root = os.path.join(project_root, "data", lang, "test", "images")
    if not (os.path.isfile(tsv) and os.path.isdir(img_root)):
        return None

    pairs = read_tsv(tsv)
    if not pairs:
        return None
    if len(pairs) > sample_n:
        pairs = rng.sample(pairs, sample_n)

    records = []  # (width, height, word_len)
    missing = 0
    for img_path, label in pairs:
        full = os.path.join(img_root, os.path.basename(img_path))
        if not os.path.isfile(full):
            full = os.path.join(img_root, img_path)
        try:
            with Image.open(full) as im:
                w, h = im.size
            records.append((w, h, len(label)))
        except (FileNotFoundError, OSError):
            missing += 1
    return records, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--languages", nargs="+", default=None,
                     help="Languages to include. Default: every language under data/ with a test set.")
    ap.add_argument("--sample", type=int, default=500,
                     help="Images sampled per language (default 500; use 5000 for the full test set)")
    ap.add_argument("--detail", nargs="+", default=["gujarati"],
                     help="Language(s) to break down by word-length bucket (default: gujarati)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.languages:
        langs = args.languages
    else:
        data_dir = os.path.join(args.project_root, "data")
        langs = sorted(d for d in os.listdir(data_dir)
                       if os.path.isfile(os.path.join(data_dir, d, "test", "test.tsv")))

    print("=" * 78)
    print(" 1. CROSS-LANGUAGE TEST-CROP DIMENSIONS")
    print("=" * 78)
    print(f"{'Language':<12s} {'n':>5s} {'avg WxH':>12s} {'median W':>9s} {'min W':>6s} "
          f"{'% under 50px':>13s} {'% under 100px':>14s}")
    print("-" * 78)

    all_records = {}
    for lang in langs:
        result = analyze_language(lang, args.project_root, args.sample, rng)
        if result is None:
            print(f"{lang:<12s}  (no test data found — skipped)")
            continue
        records, missing = result
        if not records:
            print(f"{lang:<12s}  (no readable images — skipped, {missing} missing)")
            continue
        all_records[lang] = records
        ws = [r[0] for r in records]
        hs = [r[1] for r in records]
        under50 = 100 * sum(1 for w in ws if w < 50) / len(ws)
        under100 = 100 * sum(1 for w in ws if w < 100) / len(ws)
        note = f"  ({missing} unreadable)" if missing else ""
        print(f"{lang:<12s} {len(records):>5d} {sum(ws)/len(ws):>6.0f}x{sum(hs)/len(hs):<5.0f} "
              f"{median(ws):>9.0f} {min(ws):>6d} {under50:>12.1f}% {under100:>13.1f}%{note}")

    for lang in args.detail:
        if lang not in all_records:
            continue
        print()
        print("=" * 78)
        print(f" 2. {lang.upper()}: CROP WIDTH BY WORD-LENGTH BUCKET")
        print("    (compare against the language's per-bucket WRR from eval analysis)")
        print("=" * 78)
        by_bucket = {}
        for w, h, ln in all_records[lang]:
            by_bucket.setdefault(bucket(ln), []).append(w)
        print(f"{'length':<8s} {'n':>5s} {'avg width':>10s} {'median width':>13s} {'% under 50px':>13s}")
        print("-" * 55)
        for b in BUCKET_ORDER:
            ws = by_bucket.get(b)
            if not ws:
                continue
            under50 = 100 * sum(1 for w in ws if w < 50) / len(ws)
            print(f"{b:<8s} {len(ws):>5d} {sum(ws)/len(ws):>10.0f} {median(ws):>13.0f} {under50:>12.1f}%")

    print()
    print("How to read this: the model letterboxes every crop up to 224px input.")
    print("A crop under ~50px wide is heavily upscaled and may be illegible even to a")
    print("perfect model. If one language's crops are systematically smaller — or if,")
    print("within a language, the worst-accuracy length bucket also has the smallest")
    print("crops — image size (a data property) is the likely driver, not the model.")


if __name__ == "__main__":
    main()
