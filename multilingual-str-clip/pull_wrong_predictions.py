#!/usr/bin/env python3
"""
pull_wrong_predictions.py

Copies every (or a capped sample of) wrongly-predicted image from a
evaluate.py --out_csv predictions file, organized into subfolders by error
category, plus a manifest.csv per category with GT/prediction for reference
while browsing.

Categories:
  - zwj_artifact:       prediction matches GT exactly once invisible
                         ZWJ/ZWNJ/ZWSP characters are stripped from GT.
                         (the model got every visible character right)
  - severe_truncation:  prediction length <= 1/3 of GT length (GT >= 4 chars)
  - other_wrong:        every other wrong prediction (near-misses, garbled
                         output, etc.)

Usage:
    python3 pull_wrong_predictions.py \
        --predictions_csv results/bstd_hindi_predictions.csv \
        --image_root ../bstd/Recognition/test/hindi \
        --out_dir bstd_wrong_review \
        --cap 200
"""
import csv
import argparse
import shutil
from pathlib import Path

INVISIBLE = set('\u200b\u200c\u200d')


def strip_invisible(s):
    return ''.join(c for c in s if c not in INVISIBLE)


def categorize(gt, pred):
    gt_visible = strip_invisible(gt)
    if pred == gt_visible and pred != gt:
        return 'zwj_artifact'
    if len(pred) <= max(1, len(gt_visible) // 3) and len(gt_visible) >= 4:
        return 'severe_truncation'
    return 'other_wrong'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions_csv', required=True)
    ap.add_argument('--image_root', required=True,
                     help='Folder the image_path column in predictions_csv is relative to')
    ap.add_argument('--out_dir', default='wrong_review')
    ap.add_argument('--cap', type=int, default=0,
                     help='Max images to copy PER CATEGORY. 0 = no cap, copy everything.')
    args = ap.parse_args()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    with open(args.predictions_csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    by_category = {'zwj_artifact': [], 'severe_truncation': [], 'other_wrong': []}
    missing_images = []

    for r in rows:
        if r.get('exact_match', 'True') == 'True':
            continue
        cat = categorize(r['ground_truth'], r['prediction'])
        by_category[cat].append(r)

    print(f"Total predictions: {len(rows)}")
    print(f"Total wrong: {sum(len(v) for v in by_category.values())}")
    for cat, entries in by_category.items():
        print(f"  {cat:20s}: {len(entries)}")
    print()

    for cat, entries in by_category.items():
        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        to_copy = entries if args.cap == 0 else entries[:args.cap]
        manifest_rows = []

        for r in to_copy:
            src = image_root / r['image_path']
            if not src.exists():
                missing_images.append(str(src))
                continue
            dst = cat_dir / r['image_path']
            shutil.copy(src, dst)
            manifest_rows.append(r)

        manifest_path = cat_dir / 'manifest.csv'
        with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'ground_truth', 'prediction'])
            writer.writeheader()
            for r in manifest_rows:
                writer.writerow({'image_path': r['image_path'],
                                  'ground_truth': r['ground_truth'],
                                  'prediction': r['prediction']})

        print(f"{cat}: copied {len(manifest_rows)} image(s) "
              f"{'(capped)' if args.cap and len(entries) > args.cap else '(all)'} to {cat_dir}")

    if missing_images:
        print(f"\n!! {len(missing_images)} image(s) referenced in the CSV were not found under --image_root:")
        for p in missing_images[:5]:
            print(f"     {p}")

    print(f"\nDone. Review folders under: {out_dir}")
    print("Each category folder has a manifest.csv alongside the images for reference.")


if __name__ == '__main__':
    main()
