#!/usr/bin/env python3
"""
organize_fonts.py
Extracts a language's font archive (.zip, .tar.gz, .tgz, .tar — or an
already-extracted folder) and copies every font file found anywhere
within it — regardless of how deeply nested (Google Fonts downloads
typically put each family in its own subfolder, often with a further
static/ subfolder inside that) — flat into fonts/<language>/. Non-font
files (OFL.txt, README.txt, etc.) are skipped automatically.

Usage — simplest case, drop the archive in incoming/ and go:
    cp punjabi_fonts.zip incoming/fonts_punjabi.zip
    python organize_fonts.py --language punjabi --dry_run
    python organize_fonts.py --language punjabi

Or point at any archive/folder directly:
    python organize_fonts.py --language punjabi --archive ~/Downloads/some_fonts.zip
    python organize_fonts.py --language punjabi --extracted_dir ~/Downloads/already_unzipped_fonts/
"""
import argparse
import os
import shutil
import tarfile
import zipfile
from collections import defaultdict

FONT_EXTS = (".ttf", ".otf", ".ttc")
ARCHIVE_EXTS = [".zip", ".tar.gz", ".tgz", ".tar"]


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
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                tf.extractall(dest_dir)
    else:
        raise SystemExit(f"Unrecognized archive type (expected one of {ARCHIVE_EXTS}): {archive_path}")

    n_files = sum(len(files) for _, _, files in os.walk(dest_dir))
    print(f"Extracted {n_files} file(s).")


def find_fonts_recursive(directory):
    found = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(FONT_EXTS):
                found.append(os.path.join(root, f))
    return found


def resolve_source_dir(args):
    if args.extracted_dir:
        if not os.path.isdir(args.extracted_dir):
            raise SystemExit(f"--extracted_dir not found: {args.extracted_dir}")
        return args.extracted_dir

    incoming_dir = os.path.join(args.project_root, "incoming")
    target_dir = os.path.join(incoming_dir, f"fonts_{args.language}_extracted")

    archive = args.archive
    if not archive:
        checked = []
        for ext in ARCHIVE_EXTS:
            candidate = os.path.join(incoming_dir, f"fonts_{args.language}{ext}")
            checked.append(candidate)
            if os.path.isfile(candidate):
                archive = candidate
                break
        if not archive:
            checked_lines = "\n".join(f"  {c}: not found" for c in checked)
            raise SystemExit(
                f"No font archive found for '{args.language}'. Checked:\n{checked_lines}\n"
                f"Pass --archive <path> explicitly, or place one at "
                f"incoming/fonts_{args.language}.zip (or .tar.gz/.tgz/.tar)."
            )
    elif not os.path.isfile(archive):
        raise SystemExit(f"--archive not found: {archive}")

    extract_archive(archive, target_dir)
    return target_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--archive", default=None,
                     help="Path to a font archive (.zip/.tar.gz/.tgz/.tar). If neither this nor "
                          "--extracted_dir is given, incoming/fonts_<language>.{zip,tar.gz,tgz,tar} "
                          "is auto-detected.")
    ap.add_argument("--extracted_dir", default=None,
                     help="An already-extracted folder to organize fonts from, instead of an archive.")
    ap.add_argument("--project_root", default=".", help="multilingual-str-clip project root")
    ap.add_argument("--dry_run", action="store_true", help="Preview what would be copied — copies nothing")
    args = ap.parse_args()

    source_dir = resolve_source_dir(args)

    fonts = find_fonts_recursive(source_dir)
    if not fonts:
        raise SystemExit(f"No font files (.ttf/.otf/.ttc) found anywhere under {source_dir}.")

    by_basename = defaultdict(list)
    for f in fonts:
        by_basename[os.path.basename(f)].append(f)
    collisions = {k: v for k, v in by_basename.items() if len(v) > 1}
    if collisions:
        print(f"!! {len(collisions)} filename collision(s) found — these would overwrite each other "
              f"in the flat destination folder. First few:")
        for name, paths in list(collisions.items())[:5]:
            print(f"     {name}:")
            for p in paths:
                print(f"       {p}")
        raise SystemExit(1)

    dest_dir = os.path.join(args.project_root, "fonts", args.language)
    print(f"\nFound {len(fonts)} font file(s) under {source_dir}")
    print(f"Destination: {dest_dir}/\n")

    families = defaultdict(int)
    for f in fonts:
        rel = os.path.relpath(f, source_dir)
        top = rel.split(os.sep)[0]
        families[top] += 1
    print("By family/folder found:")
    for fam, count in sorted(families.items()):
        print(f"  {fam}: {count} font(s)")

    if not args.dry_run:
        os.makedirs(dest_dir, exist_ok=True)
        for f in fonts:
            shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))
        print(f"\nCopied {len(fonts)} font file(s) to {dest_dir}")
    else:
        print(f"\n(dry run — would copy {len(fonts)} font file(s) to {dest_dir}. Re-run without "
              f"--dry_run to actually copy.)")


if __name__ == "__main__":
    main()
