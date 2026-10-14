#!/usr/bin/env python3
"""
append_to_demo_gallery.py
Appends one live-demo result (image + language + predicted text) to a
persistent HTML gallery page, newest result first, so a sequence of demo
runs during a presentation accumulate into one browsable page instead of
each overwriting the last or only appearing in a terminal.

Called by demo.sh -- not normally run by hand, but safe to:
    python append_to_demo_gallery.py --language telugu \
        --image_path /abs/path/to/image.jpg --prediction "కొండాపూర్" \
        --gallery_path demo/gallery.html
"""
import argparse
import html
import os
import shutil
from datetime import datetime

PAGE_TMPL = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Demo Gallery</title>
<style>
  body {{ font-family: sans-serif; margin: 28px; background: #fafafa; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #666; margin-bottom: 24px; }}
  .grid {{ display: flex; flex-direction: column; gap: 18px; }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 10px;
           padding: 18px 22px; display: flex; align-items: center; gap: 26px; }}
  .card img {{ max-width: 260px; max-height: 200px; border: 1px solid #eee;
               border-radius: 6px; flex-shrink: 0; }}
  .info {{ flex: 1; }}
  .badge {{ display: inline-block; background: #1a3a6b; color: white;
            font-size: 12px; padding: 3px 10px; border-radius: 12px;
            text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 10px; }}
  .prediction {{ font-size: 40px; font-weight: 700; color: #111; margin: 6px 0; }}
  .timestamp {{ color: #999; font-size: 12px; margin-top: 8px; }}
  .empty-warn {{ color: #b42318; font-weight: 600; }}
</style>
</head>
<body>
<h1>Live Demo \u2014 Multilingual Scene Text Recognition</h1>
<div class="sub">Most recent result first. This page updates automatically as new demos run.</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

CARD_TMPL = """<div class="card">
<img src="{img_src}" alt="">
<div class="info">
<span class="badge">{lang}</span>
<div class="prediction{empty_class}">{prediction}</div>
<div class="timestamp">{timestamp} &middot; {fname}</div>
</div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--image_path", required=True, help="Absolute path to the demo image")
    ap.add_argument("--prediction", default="", help="Predicted text (may be empty on failure)")
    ap.add_argument("--gallery_path", default=None,
                     help="Path to the gallery HTML file (default: <project_root>/demo/gallery.html)")
    ap.add_argument("--project_root", default=".")
    args = ap.parse_args()

    gallery_path = args.gallery_path or os.path.join(args.project_root, "demo", "gallery.html")
    gallery_dir = os.path.dirname(gallery_path)
    images_dir = os.path.join(gallery_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Copy the image into the gallery's own images/ folder with a unique,
    # timestamped name -- so the gallery keeps working even if the original
    # demo image is later overwritten (demo.sh reuses one filename per
    # language) or deleted.
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    ts_fname = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    orig_ext = os.path.splitext(args.image_path)[1] or ".jpg"
    stored_fname = f"{ts_fname}_{args.language}{orig_ext}"
    stored_path = os.path.join(images_dir, stored_fname)
    if os.path.isfile(args.image_path):
        shutil.copy2(args.image_path, stored_path)
    else:
        raise SystemExit(f"Image not found, nothing to add to gallery: {args.image_path}")

    prediction = args.prediction.strip()
    if prediction:
        pred_display = html.escape(prediction)
        empty_class = ""
    else:
        pred_display = "(no prediction \u2014 check terminal output for an error)"
        empty_class = " empty-warn"

    new_card = CARD_TMPL.format(
        img_src=html.escape(os.path.join("images", stored_fname)),
        lang=html.escape(args.language),
        prediction=pred_display,
        empty_class=empty_class,
        timestamp=ts_str,
        fname=html.escape(os.path.basename(args.image_path)),
    )

    # Prepend to existing cards (newest first) if the gallery already exists.
    existing_cards = ""
    if os.path.isfile(gallery_path):
        with open(gallery_path, encoding="utf-8") as f:
            existing = f.read()
        start = existing.find('<div class="grid">')
        end = existing.find('</div>\n</body>')
        if start != -1 and end != -1:
            inner_start = existing.find(">", start) + 1
            existing_cards = existing[inner_start:end].strip()

    combined = new_card if not existing_cards else new_card + "\n" + existing_cards

    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(PAGE_TMPL.format(cards=combined))

    print(f"Gallery updated: {gallery_path}")


if __name__ == "__main__":
    main()
