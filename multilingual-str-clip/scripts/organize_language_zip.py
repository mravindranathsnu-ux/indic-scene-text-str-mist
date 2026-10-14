#!/usr/bin/env python3
"""
organize_language_zip.py
Extracts a language's archive (.zip, .tar.gz, .tgz, or .tar — auto-detected
by extension; if not already extracted) and auto-detects train/val/test
image folders inside it (handles common naming variants: val/valid/
validation, test/testing), then copies their images into this project's
expected layout: data/<language>/<split>/images/

Usage — simplest case, just drop the archive in incoming/ and go:
    cp kannada.tar.gz incoming/
    python organize_language_zip.py --language kannada --dry_run
    python organize_language_zip.py --language kannada

Resolution order for where the data comes from (first match wins):
    1. --extracted_dir, if given (an already-extracted folder)
    2. --archive (or the older --zip, same thing), if given — extracted
       automatically to incoming/<language>/
    3. incoming/<language>.{zip,tar.gz,tgz,tar}, auto-detected — extracted
       automatically
    4. incoming/<language>/, if it already has content (already extracted)
"""
import argparse
import os
import shutil
import tarfile
import zipfile
from collections import defaultdict

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
ARCHIVE_EXTS = [".zip", ".tar.gz", ".tgz", ".tar"]  # checked in this order when auto-detecting

SPLIT_NAME_PATTERNS = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}


def strip_archive_ext(filename):
    """Returns the filename with a known archive extension removed, or None
    if it doesn't end in one of ARCHIVE_EXTS."""
    lower = filename.lower()
    for ext in ARCHIVE_EXTS:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return None


def find_split_sources(extracted_dir):
    """Walks extracted_dir and returns {split: [source_dir_paths]} — a
    'source' is either a directory whose name matches a split pattern, or
    an archive FILE whose name (minus extension) matches one, in which
    case it's extracted automatically (one level deeper) and the resulting
    folder is used as the source. Handles archives that unpack into
    train.tar.gz / val.tar.gz / test.tar.gz rather than plain folders."""
    matches = defaultdict(list)
    for root, dirs, files in os.walk(extracted_dir):
        for d in dirs:
            name_lower = d.lower()
            for split, patterns in SPLIT_NAME_PATTERNS.items():
                if name_lower in patterns:
                    matches[split].append(os.path.join(root, d))

        for f in files:
            base = strip_archive_ext(f)
            if base is None:
                continue
            base_lower = base.lower()
            for split, patterns in SPLIT_NAME_PATTERNS.items():
                if base_lower in patterns:
                    nested_archive_path = os.path.join(root, f)
                    nested_dest = os.path.join(root, f"_extracted_{base}")
                    if not os.path.isdir(nested_dest) or not os.listdir(nested_dest):
                        extract_archive(nested_archive_path, nested_dest)
                    matches[split].append(nested_dest)
    return matches


def find_images_recursive(directory):
    found = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                found.append(os.path.join(root, f))
    return found


def extract_archive(archive_path, dest_dir):
    print(f"Extracting {archive_path} -> {dest_dir} ...")
    os.makedirs(dest_dir, exist_ok=True)
    lower = archive_path.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            bad_file = zf.testzip()
            if bad_file:
                raise SystemExit(f"Corrupt zip: bad CRC on {bad_file!r} inside {archive_path}")
            zf.extractall(dest_dir)
    elif lower.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as tf:
            # filter="data" (Python 3.12+) rejects unsafe paths (e.g. ../../etc/passwd,
            # absolute paths) during extraction -- fall back gracefully on older Python
            # where tarfile.extractall doesn't accept a filter argument at all.
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                tf.extractall(dest_dir)
    else:
        raise SystemExit(f"Unrecognized archive type (expected one of {ARCHIVE_EXTS}): {archive_path}")

    n_files = sum(len(files) for _, _, files in os.walk(dest_dir))
    print(f"Extracted {n_files} file(s).")


def resolve_extracted_dir(args):
    """Returns (extracted_dir, did_extract_bool)."""
    if args.extracted_dir:
        if not os.path.isdir(args.extracted_dir):
            raise SystemExit(f"--extracted_dir not found: {args.extracted_dir}")
        return args.extracted_dir, False

    incoming_dir = os.path.join(args.project_root, "incoming")
    target_dir = os.path.join(incoming_dir, args.language)

    archive_arg = args.archive or args.zip
    if archive_arg:
        if not os.path.isfile(archive_arg):
            raise SystemExit(f"--archive not found: {archive_arg}")
        extract_archive(archive_arg, target_dir)
        return target_dir, True

    checked = []
    for ext in ARCHIVE_EXTS:
        candidate = os.path.join(incoming_dir, f"{args.language}{ext}")
        checked.append(candidate)
        if os.path.isfile(candidate):
            print(f"Auto-detected {candidate}")
            extract_archive(candidate, target_dir)
            return target_dir, True

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"Using already-extracted folder: {target_dir}")
        return target_dir, False

    checked_lines = "\n".join(f"  {c}: not found" for c in checked)
    raise SystemExit(
        f"Nothing to organize for '{args.language}'. Checked:\n"
        f"  --extracted_dir: (not given)\n"
        f"  --archive: (not given)\n"
        f"{checked_lines}\n"
        f"  {target_dir}: missing or empty\n"
        f"Put the archive at one of those locations (e.g. `cp {args.language}.tar.gz {incoming_dir}/`) "
        f"and re-run, or pass --archive/--extracted_dir explicitly."
    )


def find_label_files(directory, split):
    """Looks for <split>_images.txt / <split>_labels.txt (the raw label
    format this project's 00c_combine_images_labels.sh expects) anywhere
    under directory. Returns {'images': path_or_None, 'labels': path_or_None}."""
    found = {"images": None, "labels": None}
    target_images = f"{split}_images.txt"
    target_labels = f"{split}_labels.txt"
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f == target_images and found["images"] is None:
                found["images"] = os.path.join(root, f)
            elif f == target_labels and found["labels"] is None:
                found["labels"] = os.path.join(root, f)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted_dir", default=None,
                     help="An already-extracted folder to organize from. Takes priority over --archive.")
    ap.add_argument("--archive", default=None,
                     help="Path to the language's archive (.zip/.tar.gz/.tgz/.tar) — extracted "
                          "automatically to incoming/<language>/ before organizing. If neither this "
                          "nor --extracted_dir is given, incoming/<language>.{zip,tar.gz,tgz,tar} "
                          "is auto-detected.")
    ap.add_argument("--zip", default=None, help=argparse.SUPPRESS)  # old name, kept working
    ap.add_argument("--language", required=True)
    ap.add_argument("--project_root", default=".", help="multilingual-str-clip project root")
    ap.add_argument("--dry_run", action="store_true",
                     help="Preview what would be copied into data/<language>/ — copies nothing there. "
                          "NOTE: if an archive needs extracting, that extraction to incoming/<language>/ "
                          "still happens even in --dry_run (it's just unpacking your own archive, not "
                          "touching data/); only the final copy into data/<language>/ is skipped.")
    args = ap.parse_args()

    extracted_dir, _did_extract = resolve_extracted_dir(args)

    split_dirs = find_split_sources(extracted_dir)

    print("=" * 60)
    print(" DETECTED SPLIT FOLDERS")
    print("=" * 60)
    problems = []
    for split in ("train", "val", "test"):
        candidates = split_dirs.get(split, [])
        if len(candidates) == 0:
            print(f"  [MISSING] {split}: no folder matching {SPLIT_NAME_PATTERNS[split]} found")
            problems.append(split)
        elif len(candidates) > 1:
            print(f"  [AMBIGUOUS] {split}: found {len(candidates)} candidate folders:")
            for c in candidates:
                print(f"      {c}")
            problems.append(split)
        else:
            print(f"  [OK] {split}: {candidates[0]}")
    print("=" * 60)

    if problems:
        print(f"\n!! Could not unambiguously resolve: {problems}. Fix the folder names/structure "
              f"(or extend SPLIT_NAME_PATTERNS in this script) and re-run before proceeding.")
        raise SystemExit(1)

    dest_root = os.path.join(args.project_root, "data", args.language)
    print(f"\nDestination: {dest_root}/<split>/images/\n")

    grand_total = 0
    for split in ("train", "val", "test"):
        src_dir = split_dirs[split][0]
        images = find_images_recursive(src_dir)
        dest_dir = os.path.join(dest_root, split, "images")

        # Check for basename collisions before copying anything -- two
        # source images with the same filename would otherwise silently
        # overwrite one another in the flat destination folder.
        basenames = defaultdict(list)
        for p in images:
            basenames[os.path.basename(p)].append(p)
        collisions = {k: v for k, v in basenames.items() if len(v) > 1}
        if collisions:
            print(f"!! [{split}] {len(collisions)} filename collision(s) found — these would "
                  f"overwrite each other in the flat destination folder. First few:")
            for name, paths in list(collisions.items())[:5]:
                print(f"     {name}: {paths}")
            raise SystemExit(1)

        print(f"[{split}] {len(images)} image(s) found in {src_dir}")
        if not args.dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            for p in images:
                shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
            print(f"       -> copied to {dest_dir}")
        else:
            print(f"       -> (dry run — would copy to {dest_dir})")
        grand_total += len(images)

        # Also copy <split>_images.txt / <split>_labels.txt if present
        # anywhere under this split's source folder -- these ship bundled
        # alongside the images in this dataset's archives, but 00c
        # expects them directly in data/<language>/<split>/, not inside
        # images/, so they need copying (not just the images).
        label_files = find_label_files(src_dir, split)
        split_dest = os.path.join(dest_root, split)
        for kind, src_path in label_files.items():
            if src_path is None:
                print(f"  !! [{split}] no {split}_{kind}.txt found under {src_dir} — "
                      f"00c will report this split as skipped until it's in place.")
                continue
            dest_path = os.path.join(split_dest, os.path.basename(src_path))
            print(f"  [{split}] found {os.path.basename(src_path)} -> "
                  f"{'(dry run, would copy to) ' if args.dry_run else 'copying to '}{dest_path}")
            if not args.dry_run:
                os.makedirs(split_dest, exist_ok=True)
                shutil.copy2(src_path, dest_path)

    print(f"\n{'Would copy' if args.dry_run else 'Copied'} {grand_total} image(s) total.")
    if args.dry_run:
        print("Re-run without --dry_run to actually copy.")


if __name__ == "__main__":
    main()
