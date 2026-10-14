"""
split_annotations.py
Splits a single ground-truth annotation file (gt.txt) into train/val/test
TSVs for use with dataset.py / train.py / evaluate.py.

Produces THREE genuinely disjoint files:
  - train.tsv : used for training
  - val.tsv   : used every epoch during training to pick the best checkpoint
                (train.py saves best.pt based on val.tsv's CER) — touch this
                as often as you like while tuning hyperparameters
  - test.tsv  : held out completely — never used for training or checkpoint
                selection. Only evaluate on this once you're done tuning,
                and treat that number as the real one. Evaluating on val.tsv
                (or reusing val.tsv AS test.tsv) gives an optimistically
                biased number, since it's the exact file checkpoint
                selection was fit against.

Expected input format, one sample per line (default separator: TAB):
    image_path<SEP>label
e.g.
    images/img_0001.jpg	తెలుగు
    images/img_0002.jpg	హైదరాబాద్

Usage (three-way split):
    python split_annotations.py --gt gt.txt --out_dir data --val_ratio 0.1 --test_ratio 0.1

Usage (legacy two-way split — omit --test_ratio, or set it to 0):
    python split_annotations.py --gt gt.txt --out_dir data --val_ratio 0.1
"""
import argparse
import os
import random


def read_lines(gt_path, sep):
    samples = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if sep == " ":
                # collapse runs of whitespace (some annotation tools pad with
                # multiple spaces/tabs) instead of splitting on every single space
                parts = line.split(None, 1)
            else:
                parts = line.split(sep, 1)  # split once: filename can't contain sep, but label might
            if len(parts) < 2 or not parts[0] or not parts[1]:
                print(f"  [skip] line {line_num}: could not split on {sep!r}: {line!r}")
                continue
            img_path, label = parts[0], parts[1]
            samples.append((img_path, label))
    return samples


def write_tsv(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for img_path, label in samples:
            f.write(f"{img_path}\t{label}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to the full ground-truth file (gt.txt)")
    ap.add_argument("--out_dir", default="data", help="Where to write train.tsv / val.tsv / test.tsv")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of samples held out for validation")
    ap.add_argument("--test_ratio", type=float, default=0.0,
                     help="Fraction of samples held out as a genuinely separate test set (never used "
                          "for training or checkpoint selection). 0 (default) skips test.tsv entirely "
                          "for backward compatibility — set this (e.g. 0.1) to get a real three-way "
                          "split. val_ratio + test_ratio must be < 1.")
    ap.add_argument("--sep", default="\t", help="Field separator used in gt.txt (default: TAB). "
                                                 "Use ',' if your file is comma-separated.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for a reproducible split")
    ap.add_argument("--dedup", action="store_true",
                     help="Drop duplicate image_path entries before splitting, keeping the first "
                          "occurrence (prevents the same image landing in more than one split)")
    args = ap.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError(f"--val_ratio ({args.val_ratio}) + --test_ratio ({args.test_ratio}) must be < 1.0")

    os.makedirs(args.out_dir, exist_ok=True)

    samples = read_lines(args.gt, args.sep)
    print(f"Read {len(samples)} labeled samples from {args.gt}")

    if args.dedup:
        seen = set()
        deduped = []
        for img_path, label in samples:
            if img_path in seen:
                continue
            seen.add(img_path)
            deduped.append((img_path, label))
        if len(deduped) < len(samples):
            print(f"Dropped {len(samples) - len(deduped)} duplicate image_path entries")
        samples = deduped

    if len(samples) == 0:
        raise RuntimeError("No valid samples parsed from gt.txt — check --sep matches your file's format.")

    random.seed(args.seed)
    random.shuffle(samples)

    val_size = max(1, int(round(len(samples) * args.val_ratio)))
    test_size = int(round(len(samples) * args.test_ratio)) if args.test_ratio > 0 else 0

    val_samples = samples[:val_size]
    test_samples = samples[val_size:val_size + test_size]
    train_samples = samples[val_size + test_size:]

    train_path = os.path.join(args.out_dir, "train.tsv")
    val_path = os.path.join(args.out_dir, "val.tsv")
    write_tsv(train_path, train_samples)
    write_tsv(val_path, val_samples)
    print(f"Train: {len(train_samples)} samples -> {train_path}")
    print(f"Val:   {len(val_samples)} samples -> {val_path}")

    if test_size > 0:
        test_path = os.path.join(args.out_dir, "test.tsv")
        write_tsv(test_path, test_samples)
        print(f"Test:  {len(test_samples)} samples -> {test_path}  "
              f"(genuinely held out — only evaluate on this once, after tuning is done)")
    else:
        print("No --test_ratio given: test.tsv NOT created. If you point evaluate.py at val.tsv "
              "instead, remember that number is optimistically biased (it's the same file "
              "checkpoint selection was fit against) — re-run with --test_ratio to get a real one.")


if __name__ == "__main__":
    main()
