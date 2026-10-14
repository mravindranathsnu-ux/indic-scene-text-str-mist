#!/usr/bin/env python3
"""
combine_gujarati_datasets.py
Combines multiple Gujarati training-image sources into one train.tsv +
images/ directory, safely:
  - prefixes every image filename by its source, so two sources that
    happen to reuse the same bare filename (e.g. "img_001.jpg") never
    collide and silently overwrite each other
  - hashes every combined-train image's content and checks it against
    the fixed, trusted test set, refusing to proceed if any train image
    is byte-identical to a test image (that word's label would leak
    into training otherwise) -- same check this project's own
    05_check_duplicates.sh already runs for every other language
  - reports (but does not block on) duplicate images *within* the
    combined train pool across sources, since some overlap there is
    expected and harmless, just worth knowing about

Usage:
    python combine_gujarati_datasets.py \
        --sources data/gujarati/train:verified data/gujarati_backup_set1/train:orig17500 data/gujarati_cropped:new921 \
        --test_dir data/gujarati/test \
        --out_dir data/gujarati_setA \
        --out_tsv data/gujarati_setA/train.tsv

Each --sources entry is <directory>:<tag>. <directory> must contain both
the images and a train.tsv (or *_labels.txt + *_images.txt, whichever
this project's own 00c step already produces) in image_path<TAB>label
format. <tag> becomes the filename prefix for that source in the
combined output, and is never reused across sources you pass in the
same run (checked before anything is copied).
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path


def file_hash(path, chunk_size=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_source_tsv(source_dir):
    """Match this project's own convention: prefer an existing train.tsv,
    fall back to combining *_images.txt + *_labels.txt if that's what's
    present instead (mirrors 00c_combine_images_labels.sh's own logic)."""
    direct = os.path.join(source_dir, "train.tsv")
    if os.path.isfile(direct):
        return direct

    # Look for a TSV with any name directly in the folder
    tsvs = [f for f in os.listdir(source_dir) if f.endswith(".tsv")]
    if len(tsvs) == 1:
        return os.path.join(source_dir, tsvs[0])

    # Fall back: *_images.txt + *_labels.txt pair
    images_txt = [f for f in os.listdir(source_dir) if f.endswith("_images.txt")]
    labels_txt = [f for f in os.listdir(source_dir) if f.endswith("_labels.txt")]
    if len(images_txt) == 1 and len(labels_txt) == 1:
        img_lines = open(os.path.join(source_dir, images_txt[0]), encoding="utf-8").read().splitlines()
        lbl_lines = open(os.path.join(source_dir, labels_txt[0]), encoding="utf-8").read().splitlines()
        if len(img_lines) != len(lbl_lines):
            raise SystemExit(f"!! {source_dir}: images/labels line count mismatch "
                              f"({len(img_lines)} vs {len(lbl_lines)})")
        synthetic_tsv = os.path.join(source_dir, "_combined_from_images_labels.tsv")
        with open(synthetic_tsv, "w", encoding="utf-8") as f:
            for img, lbl in zip(img_lines, lbl_lines):
                f.write(f"{img}\t{lbl}\n")
        return synthetic_tsv

    raise SystemExit(f"!! Could not find a train.tsv (or *_images.txt/*_labels.txt pair) in {source_dir}. "
                      f"Files present: {os.listdir(source_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                     help="One or more <directory>:<tag> pairs")
    ap.add_argument("--test_dir", required=True,
                     help="Fixed, trusted test set directory (must contain images/ and test.tsv or similar)")
    ap.add_argument("--out_dir", required=True, help="Output directory for combined images/")
    ap.add_argument("--out_tsv", required=True, help="Output combined train.tsv path")
    args = ap.parse_args()

    # --- parse and validate sources ---
    parsed_sources = []
    seen_tags = set()
    for entry in args.sources:
        if ":" not in entry:
            raise SystemExit(f"!! --sources entry '{entry}' must be in <directory>:<tag> format")
        directory, tag = entry.rsplit(":", 1)
        if tag in seen_tags:
            raise SystemExit(f"!! Duplicate tag '{tag}' -- each source needs a unique tag")
        if not os.path.isdir(directory):
            raise SystemExit(f"!! Source directory not found: {directory}")
        seen_tags.add(tag)
        parsed_sources.append((directory, tag))

    print(f"Sources to combine ({len(parsed_sources)}):")
    for d, t in parsed_sources:
        print(f"  [{t}] {d}")

    # --- hash the fixed test set first ---
    test_images_dir = os.path.join(args.test_dir, "images")
    if not os.path.isdir(test_images_dir):
        raise SystemExit(f"!! No images/ subdirectory found in --test_dir {args.test_dir}")
    print(f"\nHashing test set images in {test_images_dir} ...")
    test_hashes = {}
    for fname in os.listdir(test_images_dir):
        fpath = os.path.join(test_images_dir, fname)
        if os.path.isfile(fpath):
            test_hashes[file_hash(fpath)] = fname
    print(f"  {len(test_hashes)} test image(s) hashed")

    # --- process each source: copy images with prefix, collect labels ---
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    combined_rows = []
    train_hashes = {}  # hash -> (tag, prefixed_filename) for within-train dup reporting
    leakage_found = []
    within_train_dups = 0
    total_copied = 0

    for source_dir, tag in parsed_sources:
        tsv_path = find_source_tsv(source_dir)
        # image root: prefer an images/ subfolder, else the source dir itself
        img_root_candidates = [os.path.join(source_dir, "images"), source_dir]
        img_root = next((d for d in img_root_candidates if os.path.isdir(d)), source_dir)

        rows = list(csv.reader(open(tsv_path, encoding="utf-8"), delimiter="\t"))
        print(f"\n[{tag}] {len(rows)} annotation(s) from {tsv_path}")
        print(f"[{tag}] resolving images against: {img_root}")

        copied_this_source = 0
        missing_this_source = 0
        for row in rows:
            if len(row) < 2:
                continue
            rel_path, label = row[0], "\t".join(row[1:])
            src_img_path = os.path.join(img_root, rel_path)
            if not os.path.isfile(src_img_path):
                # try bare filename directly under img_root as a fallback
                alt = os.path.join(img_root, os.path.basename(rel_path))
                if os.path.isfile(alt):
                    src_img_path = alt
                else:
                    missing_this_source += 1
                    continue

            h = file_hash(src_img_path)
            if h in test_hashes:
                leakage_found.append((tag, rel_path, test_hashes[h]))
                continue  # do not copy a test-leaking image into train

            if h in train_hashes:
                within_train_dups += 1

            prefixed_name = f"{tag}__{os.path.basename(rel_path)}"
            dest_path = os.path.join(args.out_dir, "images", prefixed_name)
            if not os.path.exists(dest_path):
                with open(src_img_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                    fdst.write(fsrc.read())
            combined_rows.append((prefixed_name, label))
            train_hashes[h] = (tag, prefixed_name)
            copied_this_source += 1

        print(f"[{tag}] copied {copied_this_source}, missing {missing_this_source}")
        total_copied += copied_this_source

    # --- write combined TSV ---
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for img, label in combined_rows:
            writer.writerow([img, label])

    print(f"\n{'='*60}")
    print(f" COMBINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total combined training samples: {total_copied}")
    print(f"Written to: {args.out_tsv}")
    print(f"Images in: {os.path.join(args.out_dir, 'images')}")
    print(f"Within-train duplicate images across sources (informational only): {within_train_dups}")

    if leakage_found:
        print(f"\n!!! TRAIN/TEST LEAKAGE DETECTED: {len(leakage_found)} image(s) in your training "
              f"sources are byte-identical to test set images. These were EXCLUDED from the "
              f"combined training set automatically. Details:")
        for tag, train_path, test_fname in leakage_found[:20]:
            print(f"    [{tag}] {train_path}  <-- identical to test image {test_fname}")
        if len(leakage_found) > 20:
            print(f"    ... and {len(leakage_found) - 20} more")
    else:
        print(f"\nNo train/test leakage detected -- 0 combined-train images matched the fixed test set.")


if __name__ == "__main__":
    main()
