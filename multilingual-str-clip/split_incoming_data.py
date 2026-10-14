#!/usr/bin/env python3
"""
split_incoming_data.py
Splits a fresh, flat folder of images + a single ground-truth file into
this project's per-language train/val/test structure:

    data/<language>/train/images/ + train.tsv (+ train_images.txt/_labels.txt)
    data/<language>/val/images/   + val.tsv   (+ val_images.txt/_labels.txt)
    data/<language>/test/images/  + test.tsv  (+ test_images.txt/_labels.txt)

Built for the Gujarati re-dataset (the original subset's images were found
unfit — tiny/blurry — on human inspection; a verified replacement set was
obtained from the research group), but works for any language.

Split ratio follows the project standard 70/10/20 (the official sets are
17,500/2,500/5,000 of 25,000). Existing data/<language>/ is BACKED UP by
rename, never deleted or overwritten in place.

Usage (from the project root):
    python split_incoming_data.py --language gujarati --dry_run   # inspect first
    python split_incoming_data.py --language gujarati             # actually split
"""
import argparse
import hashlib
import os
import random
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def parse_gt_file(path):
    """Auto-detect the separator (tab / comma / whitespace) and parse
    filename->label pairs. Returns (pairs, sep_name)."""
    raw_lines = []
    with open(path, encoding="utf-8-sig") as f:  # -sig strips a BOM if present
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.strip():
                raw_lines.append(line)
    if not raw_lines:
        raise SystemExit(f"GT file is empty: {path}")

    candidates = [("tab", "\t"), ("comma", ","), ("whitespace", None)]
    best = None
    for name, sep in candidates:
        pairs = []
        ok = 0
        for line in raw_lines:
            parts = line.split(sep, 1) if sep else line.split(None, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                pairs.append((parts[0].strip(), parts[1].strip()))
                ok += 1
        frac = ok / len(raw_lines)
        if best is None or frac > best[2]:
            best = (name, pairs, frac)
        if frac == 1.0:
            break
    name, pairs, frac = best
    if frac < 0.95:
        raise SystemExit(
            f"Could not confidently parse {path}: best separator '{name}' only parsed "
            f"{frac*100:.0f}% of lines. First 3 raw lines for inspection:\n  "
            + "\n  ".join(raw_lines[:3]))
    return pairs, name, len(raw_lines)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--source", default=None,
                     help="Folder containing the images and the GT file. "
                          "Default: incoming/<language>/<language>/")
    ap.add_argument("--gt_file", default="word_image_gt.txt",
                     help="GT filename inside the source folder (default: word_image_gt.txt)")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                     help="Report everything that WOULD happen; move/copy nothing")
    args = ap.parse_args()

    source = args.source or os.path.join(args.project_root, "incoming", args.language, args.language)
    if not os.path.isdir(source):
        raise SystemExit(f"Source folder not found: {source}")
    gt_path = os.path.join(source, args.gt_file)
    if not os.path.isfile(gt_path):
        listing = [f for f in os.listdir(source) if f.lower().endswith(".txt")]
        raise SystemExit(f"GT file not found: {gt_path}\n.txt files present: {listing}")

    # ---- Parse GT ----
    pairs, sep_name, n_lines = parse_gt_file(gt_path)
    print(f"GT file: {gt_path}")
    print(f"  Parsed {len(pairs)}/{n_lines} lines using separator: {sep_name}")
    print(f"  Sample entries:")
    for fn, label in pairs[:3]:
        print(f"    {fn!r} -> {label!r}")

    # ---- Match against files on disk ----
    disk_files = {f for f in os.listdir(source) if f.lower().endswith(IMG_EXTS)}
    print(f"\nImage files in source folder: {len(disk_files)}")

    gt_names = [fn for fn, _ in pairs]
    dup_gt = len(gt_names) - len(set(gt_names))
    if dup_gt:
        print(f"  !! {dup_gt} duplicate filename(s) in the GT file — keeping first occurrence of each.")
        seen = set()
        deduped = []
        for fn, label in pairs:
            if fn not in seen:
                seen.add(fn)
                deduped.append((fn, label))
        pairs = deduped

    matched = [(fn, label) for fn, label in pairs if fn in disk_files]
    gt_no_file = [fn for fn, _ in pairs if fn not in disk_files]
    file_no_gt = disk_files - {fn for fn, _ in pairs}
    print(f"  Matched image+label pairs: {len(matched)}")
    if gt_no_file:
        print(f"  !! GT entries with NO matching image file: {len(gt_no_file)} (first few: {gt_no_file[:3]})")
    if file_no_gt:
        print(f"  !! Image files with NO GT entry (will be EXCLUDED): {len(file_no_gt)} (first few: {sorted(file_no_gt)[:3]})")
    if not matched:
        raise SystemExit("No matched pairs — check that GT filenames match the actual files.")

    # ---- Byte-identical duplicate check within the source ----
    print("\nChecking for byte-identical duplicate images in the source...")
    by_hash = {}
    dupes = []
    for fn, label in matched:
        h = file_hash(os.path.join(source, fn))
        if h in by_hash:
            dupes.append((fn, by_hash[h]))
        else:
            by_hash[h] = fn
    if dupes:
        print(f"  !! {len(dupes)} byte-identical duplicate(s) found — EXCLUDING the later copy of each:")
        for fn, first in dupes[:5]:
            print(f"     {fn}  (identical to {first})")
        dupe_names = {fn for fn, _ in dupes}
        matched = [(fn, label) for fn, label in matched if fn not in dupe_names]
    else:
        print("  0 duplicates — clean.")

    # ---- Split ----
    rng = random.Random(args.seed)
    rng.shuffle(matched)
    n = len(matched)
    n_train = round(n * args.train_frac)
    n_val = round(n * args.val_frac)
    splits = {
        "train": matched[:n_train],
        "val": matched[n_train:n_train + n_val],
        "test": matched[n_train + n_val:],
    }
    print(f"\nSplit ({args.train_frac:.0%}/{args.val_frac:.0%}/{1-args.train_frac-args.val_frac:.0%}, seed={args.seed}):")
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")

    lang_dir = os.path.join(args.project_root, "data", args.language)
    backup_dir = lang_dir + "_backup_old"

    if args.dry_run:
        print(f"\n(dry run) Would back up existing {lang_dir} -> {backup_dir}, then write the new "
              f"structure. Re-run without --dry_run to do it.")
        return

    # ---- Backup existing data, never overwrite ----
    if os.path.isdir(lang_dir):
        if os.path.exists(backup_dir):
            raise SystemExit(
                f"Backup destination already exists: {backup_dir}\n"
                f"Move or remove it first — refusing to overwrite a previous backup.")
        shutil.move(lang_dir, backup_dir)
        print(f"\nBacked up existing data: {lang_dir} -> {backup_dir}")

    # ---- Write the new structure ----
    for name, items in splits.items():
        split_dir = os.path.join(lang_dir, name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        tsv_lines, img_lines, lbl_lines = [], [], []
        for fn, label in items:
            shutil.copy2(os.path.join(source, fn), os.path.join(img_dir, fn))
            tsv_lines.append(f"{fn}\t{label}")
            img_lines.append(fn)
            lbl_lines.append(label)
        with open(os.path.join(split_dir, f"{name}.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(tsv_lines) + "\n")
        with open(os.path.join(split_dir, f"{name}_images.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(img_lines) + "\n")
        with open(os.path.join(split_dir, f"{name}_labels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lbl_lines) + "\n")
        print(f"  Wrote {name}: {len(items)} images -> {split_dir}/")

    print(f"\nDone. New structure at {lang_dir}/ matches the project convention "
          f"(images/ + <split>.tsv + <split>_images.txt/_labels.txt).")
    print("Next steps: clear old checkpoints & synthetic data for this language, re-run the "
          "vocab coverage check, then the full pipeline (see chat notes).")


if __name__ == "__main__":
    main()
