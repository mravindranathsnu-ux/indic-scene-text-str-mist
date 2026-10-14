"""
check_image_duplicates.py
Checks for byte-identical images appearing under different filenames in
both a training annotation file and a test annotation file. Real scene-
text datasets can have overlapping/duplicate bounding-box annotations
that produce the same physical crop under two different filenames —
path-string-based train/val/test splitting (split_annotations.py) can't
catch this, since it only compares filenames, not pixel content.

Supports either a single shared image folder across splits, or separate
image sets per split (e.g. a database that ships train/val/test as
distinct folders) via --train_image_root / --test_image_root.

Usage (separate image sets per split):
    python check_image_duplicates.py \
        --train_image_root data/hindi/train/images \
        --test_image_root data/hindi/test/images \
        --train_ann data/hindi/train/train_augmented.tsv \
        --test_ann data/hindi/test/test.tsv

Usage (single shared image folder):
    python check_image_duplicates.py \
        --image_root data/hindi/images \
        --train_ann data/hindi/train_augmented.tsv \
        --test_ann data/hindi/test.tsv
"""
import argparse
import hashlib
import os


def read_paths(path):
    paths = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                paths.append(parts[0])
    return paths


def hash_file(image_root, path):
    full = os.path.join(image_root, path)
    if not os.path.isfile(full):
        return None
    with open(full, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", default=None,
                     help="Shared image root for both --train_ann and --test_ann. Used as a "
                          "fallback for whichever of --train_image_root/--test_image_root isn't given.")
    ap.add_argument("--train_image_root", default=None,
                     help="Folder --train_ann's image paths are relative to. Defaults to --image_root.")
    ap.add_argument("--test_image_root", default=None,
                     help="Folder --test_ann's image paths are relative to. Defaults to --image_root.")
    ap.add_argument("--train_ann", required=True,
                     help="Training annotation TSV (e.g. train_augmented.tsv) — synthetic image "
                          "paths that don't resolve under the train image root are simply skipped")
    ap.add_argument("--test_ann", required=True)
    args = ap.parse_args()

    train_image_root = args.train_image_root or args.image_root
    test_image_root = args.test_image_root or args.image_root
    if not train_image_root:
        raise ValueError("Provide --train_image_root (or --image_root as a shared fallback).")
    if not test_image_root:
        raise ValueError("Provide --test_image_root (or --image_root as a shared fallback).")

    train_paths = read_paths(args.train_ann)
    test_paths = read_paths(args.test_ann)

    train_hashes = {}
    for p in train_paths:
        h = hash_file(train_image_root, p)
        if h:
            train_hashes.setdefault(h, []).append(p)

    test_hashes = {}
    for p in test_paths:
        h = hash_file(test_image_root, p)
        if h:
            test_hashes.setdefault(h, []).append(p)

    overlap = set(train_hashes) & set(test_hashes)
    print(f"Real train images hashed: {len(train_hashes)} unique content-hashes")
    print(f"Test images hashed:       {len(test_hashes)} unique content-hashes")
    print(f"BYTE-IDENTICAL images appearing in both (different filenames possible): {len(overlap)}")
    for h in list(overlap)[:10]:
        print(f"  train: {train_hashes[h]}  <->  test: {test_hashes[h]}")


if __name__ == "__main__":
    main()
