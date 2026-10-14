#!/usr/bin/env python3
"""
crop_gujarati_words.py
Crops individual word images from full scene images using the 4-point
polygon coordinates in each image's .txt annotation file (ICDAR-style:
x1,y1,x2,y2,x3,y3,x4,y4,script,transcription per line).

Uses the axis-aligned bounding box of each polygon (min/max of the 4
points) rather than a perspective-warped crop -- validated by visual
inspection against guj_1.jpeg before being applied at scale; simple
bbox crop was sufficient since the sample polygons were only mildly
non-rectangular, not steeply rotated.

Usage:
    python crop_gujarati_words.py --input_dir extracted/Gujarati \
        --out_dir cropped_words --out_ann word_annotations.tsv
"""
import argparse
import csv
import os
from pathlib import Path
from PIL import Image


def find_image_file(input_dir, stem):
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        candidate = os.path.join(input_dir, f"{stem}.{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_annotation_line(line):
    parts = line.rstrip("\n").split(",")
    if len(parts) < 10:
        return None
    # Anchor on the script tag itself rather than assuming exactly 8
    # coordinate values (4 points) -- LabelMe polygons can have more than
    # 4 points, and at least one line in this dataset genuinely does
    # (5 points = 10 coordinate values), which silently corrupted both the
    # script field and the label when a fixed field count was assumed.
    try:
        script_idx = parts.index("Gujarati")
    except ValueError:
        return None
    coord_parts = parts[:script_idx]
    if len(coord_parts) < 4 or len(coord_parts) % 2 != 0:
        return None
    coords = list(map(float, coord_parts))
    script = parts[script_idx]
    transcription = ",".join(parts[script_idx + 1:])
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max, script, transcription


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with <stem>.{jpg,jpeg,png} + <stem>.txt files")
    ap.add_argument("--out_dir", required=True, help="Directory to write cropped word images")
    ap.add_argument("--out_ann", required=True, help="Output TSV: image_path<TAB>label")
    ap.add_argument("--min_size", type=int, default=3, help="Skip crops with width or height below this many pixels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(Path(args.input_dir).glob("*.txt"))
    print(f"Found {len(txt_files)} annotation files in {args.input_dir}")

    total_words = 0
    skipped_degenerate = 0
    skipped_no_image = 0
    per_image_counts = {}

    with open(args.out_ann, "w", encoding="utf-8", newline="") as ann_out:
        # lineterminator="\n" explicitly, not csv's RFC4180 default "\r\n" --
        # this project's dataset.py reads lines with rstrip("\n") only, so a
        # \r\n-terminated file would leave a stray \r stuck on every label.
        writer = csv.writer(ann_out, delimiter="\t", lineterminator="\n")

        for txt_path in txt_files:
            stem = txt_path.stem
            img_path = find_image_file(args.input_dir, stem)
            if img_path is None:
                print(f"  !! No image found for {stem} -- skipping its {sum(1 for _ in open(txt_path, encoding='utf-8'))} annotation(s)")
                skipped_no_image += 1
                continue

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            with open(txt_path, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]

            count_this_image = 0
            for i, line in enumerate(lines):
                parsed = parse_annotation_line(line)
                if parsed is None:
                    print(f"  !! Malformed line in {stem}.txt: {line!r}")
                    continue
                x_min, y_min, x_max, y_max, script, transcription = parsed

                # Clip to image bounds -- some coordinate sets may slightly
                # overshoot the image edge.
                x_min = max(0, min(x_min, img_w))
                x_max = max(0, min(x_max, img_w))
                y_min = max(0, min(y_min, img_h))
                y_max = max(0, min(y_max, img_h))

                w, h = x_max - x_min, y_max - y_min
                if w < args.min_size or h < args.min_size:
                    skipped_degenerate += 1
                    continue

                crop = img.crop((x_min, y_min, x_max, y_max))
                out_name = f"{stem}_{i:03d}.png"
                crop.save(os.path.join(args.out_dir, out_name))
                writer.writerow([out_name, transcription])
                total_words += 1
                count_this_image += 1

            per_image_counts[stem] = count_this_image

    print(f"\nWrote {total_words} word crop(s) to {args.out_dir}/")
    print(f"Annotation file: {args.out_ann}")
    print(f"Skipped (degenerate/too small): {skipped_degenerate}")
    print(f"Skipped (no matching image): {skipped_no_image}")
    print(f"Images processed: {len(per_image_counts)}")
    if per_image_counts:
        avg = total_words / len(per_image_counts)
        print(f"Average words per image: {avg:.1f}")


if __name__ == "__main__":
    main()
