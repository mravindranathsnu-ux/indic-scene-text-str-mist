#!/usr/bin/env python3
"""
make_font_contact_sheet.py
Combines every rendered font-check PNG (from 03_verify_fonts.sh's output,
font_check_samples/<language>/) into a single grid image, labeled with
each font's filename, so you can review 100+ fonts at a glance instead of
opening each one individually.

Usage:
    python make_font_contact_sheet.py --language punjabi
    python make_font_contact_sheet.py --language punjabi --cols 8
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--cols", type=int, default=6, help="Number of columns in the grid")
    ap.add_argument("--cell_w", type=int, default=260)
    ap.add_argument("--cell_h", type=int, default=110)
    ap.add_argument("--out", default=None,
                     help="Output path. Defaults to font_check_samples/<language>_contact_sheet.png")
    args = ap.parse_args()

    src_dir = os.path.join(args.project_root, "font_check_samples", args.language)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Not found: {src_dir} — run 03_verify_fonts.sh {args.language} first.")

    images = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    if not images:
        raise SystemExit(f"No PNGs found in {src_dir}.")

    out_path = args.out or os.path.join(src_dir, f"../{args.language}_contact_sheet.png")
    out_path = os.path.normpath(out_path)

    cols = args.cols
    rows = (len(images) + cols - 1) // cols
    cell_w, cell_h = args.cell_w, args.cell_h
    label_h = 20
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        label_font = ImageFont.load_default()

    for i, fname in enumerate(images):
        col, row = i % cols, i // cols
        x0, y0 = col * cell_w, row * (cell_h + label_h)

        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        img.thumbnail((cell_w - 10, cell_h - 10))
        px = x0 + (cell_w - img.width) // 2
        py = y0 + (cell_h - img.height) // 2
        sheet.paste(img, (px, py))

        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h + label_h - 1], outline="lightgray")
        label = fname[:36] + ("…" if len(fname) > 36 else "")
        draw.text((x0 + 5, y0 + cell_h + 2), label, fill="black", font=label_font)

    sheet.save(out_path)
    print(f"Contact sheet saved: {out_path}")
    print(f"{len(images)} font(s) in a {cols}-column grid ({rows} rows).")
    print("Scan for: garbled shapes, boxes/tofu (missing glyph), disconnected marks, "
          "or obviously wrong script — those are the ones worth opening individually.")


if __name__ == "__main__":
    main()
