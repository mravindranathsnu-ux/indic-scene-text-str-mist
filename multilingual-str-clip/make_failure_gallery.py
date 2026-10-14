#!/usr/bin/env python3
"""
make_failure_gallery.py
Builds a self-contained HTML page showing a sample of FAILING test
predictions alongside their actual crop images, so a human can judge:

  (a) Is the crop legible at all?
  (b) Is the ground-truth label actually correct?
  (c) Is the model's prediction arguably right and the label wrong?

This discriminates the remaining Gujarati hypotheses (label noise vs.
genuine illegibility vs. model deficiency) in a way no aggregate metric
can. HTML (not a rendered image grid) so the captions use YOUR machine's
fonts — the browser renders Gujarati/Odia/etc. natively.

Usage (from the project root):
    python make_failure_gallery.py --language gujarati \
        --predictions results/gujarati_test_predictions.csv
    # then open failure_gallery_gujarati.html in a browser

Options:
    --max_len 4        only failures with ground truth <= N chars (default 4,
                       since short words are Gujarati's problem zone; use 0
                       for all lengths)
    --n 60             how many failures to sample (default 60)
    --include_correct  also show a strip of CORRECT predictions for
                       comparison (helps calibrate what "normal" looks like)
"""
import argparse
import csv
import html
import os
import random


PAGE_TMPL = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Failure gallery: {lang}</title>
<style>
  body {{ font-family: sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ font-size: 20px; }}
  .note {{ color: #555; max-width: 900px; line-height: 1.4; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 18px; }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 6px;
           padding: 10px; width: 240px; }}
  .card img {{ image-rendering: pixelated; border: 1px solid #eee;
               max-width: 220px; }}
  .card .dims {{ color: #999; font-size: 11px; }}
  .lbl {{ font-size: 22px; margin: 4px 0; }}
  .gt {{ color: #1a7f37; }}
  .pred {{ color: #b42318; }}
  .lbl .tag {{ font-size: 11px; color: #888; margin-right: 6px; }}
  .sect {{ margin-top: 34px; border-top: 2px solid #ccc; padding-top: 10px; }}
</style>
</head>
<body>
<h1>Failure gallery: {lang} ({n_shown} of {n_total} failures{len_note})</h1>
<p class="note">For each card: the raw crop (shown pixel-for-pixel where possible),
its on-disk size, the ground-truth label (green) and the model's prediction (red).
Judge each: is the crop legible? is the green label actually what the sign says?
is the red prediction ever the more correct reading? A high rate of wrong/dubious
green labels points to annotation noise; clean legible crops with correct labels
that the model still misses point back at the model or a train/test mismatch.</p>
<div class="grid">
{cards}
</div>
{correct_section}
</body>
</html>
"""

CARD_TMPL = """<div class="card">
<img src="{img_src}" alt="">
<div class="dims">{fname} &middot; {w}&times;{h}px</div>
<div class="lbl gt"><span class="tag">GT</span>{gt}</div>
<div class="lbl pred"><span class="tag">PRED</span>{pred}</div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--predictions", required=True, help="Path to <lang>_test_predictions.csv")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--max_len", type=int, default=4,
                     help="Only failures with GT length <= this (0 = all lengths)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--include_correct", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_root = os.path.join(args.project_root, "data", args.language, "test", "images")
    if not os.path.isdir(img_root):
        raise SystemExit(f"Image folder not found: {img_root}")

    rows = list(csv.DictReader(open(args.predictions, encoding="utf-8")))
    fails = [r for r in rows if r["exact_match"] == "False"]
    if args.max_len > 0:
        fails = [r for r in fails if len(r["ground_truth"]) <= args.max_len]
        len_note = f", GT length \u2264 {args.max_len}"
    else:
        len_note = ""

    rng = random.Random(args.seed)
    sample = rng.sample(fails, min(args.n, len(fails)))

    def card_for(r):
        fname = os.path.basename(r["image_path"])
        full = os.path.join(img_root, fname)
        w = h = "?"
        try:
            from PIL import Image
            with Image.open(full) as im:
                w, h = im.size
        except Exception:
            pass
        # relative src so the HTML works when placed in the project root
        rel_src = os.path.join("data", args.language, "test", "images", fname)
        return CARD_TMPL.format(
            img_src=html.escape(rel_src), fname=html.escape(fname), w=w, h=h,
            gt=html.escape(r["ground_truth"]), pred=html.escape(r["prediction"] or "(empty)"),
        )

    cards = "\n".join(card_for(r) for r in sample)

    correct_section = ""
    if args.include_correct:
        corrects = [r for r in rows if r["exact_match"] == "True"]
        if args.max_len > 0:
            corrects = [r for r in corrects if len(r["ground_truth"]) <= args.max_len]
        csample = rng.sample(corrects, min(24, len(corrects)))
        correct_section = (
            '<div class="sect"><h1>For comparison: CORRECT predictions</h1>'
            '<div class="grid">' + "\n".join(card_for(r) for r in csample) + "</div></div>"
        )

    out_path = os.path.join(args.project_root, f"failure_gallery_{args.language}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(PAGE_TMPL.format(
            lang=args.language, n_shown=len(sample), n_total=len(fails),
            len_note=len_note, cards=cards, correct_section=correct_section,
        ))
    print(f"Wrote {out_path}  ({len(sample)} failure cards"
          + (", + correct-comparison strip" if args.include_correct else "") + ")")
    print(f"Open it in a browser (it references images relative to the project root).")


if __name__ == "__main__":
    main()
