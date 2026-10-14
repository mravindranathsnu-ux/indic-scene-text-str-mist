#!/usr/bin/env python3
"""
repair_manifest.py

One-off recovery for the truncation caused by the pre-fix
run_vlm_baselines.py checkpoint() bug (Sep 19 2026): a GPT-4V run
crashed mid-checkpoint on a stray UTF-16 surrogate character, and
since checkpoint() opened manifest.csv in "w" mode (truncate) and
wrote it out linearly, the crash left manifest.csv with only 1437 of
its original 3000 rows -- silently dropping the OTHER ~1563 rows'
already-completed Tesseract/PaddleOCR/EasyOCR/Google-Vision/Claude
results too, not just GPT-4V's.

This does NOT touch any crop image files (those were written once,
during the original `--stage extract` run, and were never touched
again -- they're all still on disk, untouched by this crash). It
also does NOT call any paid API. It only:

  1. Deterministically re-derives the full, original ~3000-row
     candidate list (same seed=42, same sampling code as
     build_mist_recognition_gt.py's stage_extract) to recover each
     missing row's instance_id/language/image_stem/crop_path/points.
  2. Loads the CURRENT (truncated) manifest.csv and keeps every row
     that's still in it EXACTLY as-is -- all completed OCR results
     for those 1437 rows are fully preserved, untouched.
  3. For rows missing from the current manifest, adds them back with
     every engine's status column reset to "pending" (empty text) --
     there's no way to recover text a crashed write never got to
     disk with, so those rows need each baseline re-run on them
     specifically. The resumable design of every other script means
     this only touches the recovered rows, not all 3000 --
     Tesseract/PaddleOCR/EasyOCR are free/local, GCV labeling for
     just these rows is the one step with a real (roughly
     proportional, ~1563/3000 of the original) API cost.
  4. Writes the merged, full manifest back atomically (same tmp-file
     + os.replace pattern as the other fixed scripts).

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    python3 repair_manifest.py
"""
import csv
import json
import os
import random

import cv2

import build_mist_recognition_gt as B

MANIFEST_PATH = B.MANIFEST_PATH
PER_LANGUAGE = 300
SEED = 42


def rebuild_candidate_rows():
    """Exactly reproduces stage_extract's deterministic sampling to
    recover instance_id/language/image_stem/crop_path/points for
    every one of the original ~3000 instances -- without writing any
    crop files (they already exist) or touching manifest.csv."""
    by_lang = B.load_instances()
    random.seed(SEED)
    candidates = []
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        k = min(PER_LANGUAGE, len(pool))
        sample = random.sample(pool, k)
        lang_dir = os.path.join(B.OUT_ROOT, lang)
        img_cache = {}
        for i, inst in enumerate(sample):
            img = img_cache.get(inst["image_path"])
            if img is None:
                img = cv2.imread(inst["image_path"])
                img_cache[inst["image_path"]] = img
            if img is None:
                continue
            crop = B.crop_instance(img, inst["points"])
            if crop is None or crop.size == 0:
                continue
            instance_id = f"{lang}_{i:04d}_{inst['image_stem']}_{inst['shape_idx']}"
            crop_path = os.path.join(lang_dir, instance_id + ".png")
            candidates.append({
                "instance_id": instance_id,
                "language": lang,
                "image_stem": inst["image_stem"],
                "crop_path": crop_path,
                "points": json.dumps(inst["points"]),
            })
    return candidates


def main():
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"No manifest at {MANIFEST_PATH} -- nothing to repair.")

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        existing_rows = list(csv.DictReader(f))
    if not existing_rows:
        raise SystemExit("Current manifest is empty -- can't safely repair. Stop and ask before proceeding.")

    fieldnames = list(existing_rows[0].keys())
    existing_by_id = {r["instance_id"]: r for r in existing_rows}
    print(f"Current manifest: {len(existing_rows)} rows, columns: {fieldnames}")

    candidates = rebuild_candidate_rows()
    print(f"Deterministically re-derived {len(candidates)} original candidate rows.")

    merged = []
    n_kept, n_recovered = 0, 0
    missing_crop_files = 0
    for cand in candidates:
        iid = cand["instance_id"]
        if iid in existing_by_id:
            merged.append(existing_by_id[iid])
            n_kept += 1
        else:
            row = {fn: "" for fn in fieldnames}
            row.update(cand)
            for fn in fieldnames:
                if fn == "status" or fn.endswith("_status"):
                    row[fn] = "pending"
            merged.append(row)
            n_recovered += 1
            if not os.path.exists(cand["crop_path"]):
                missing_crop_files += 1

    print(f"\nKept {n_kept} existing rows untouched (all their OCR results preserved).")
    print(f"Reconstructed {n_recovered} missing rows (reset to pending -- need re-running).")
    if missing_crop_files:
        print(f"[warn] {missing_crop_files} reconstructed rows' crop PNG files are "
              f"ALSO missing from disk -- those specific instances cannot be "
              f"recovered by this script and will need re-extraction.")
    print(f"Total after repair: {len(merged)} rows (should be ~3000).")

    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    os.replace(tmp_path, MANIFEST_PATH)
    print(f"\nRepaired manifest written: {MANIFEST_PATH}")
    print("Next: re-run each baseline script as usual (all resumable) -- "
          "they'll only touch the reconstructed 'pending' rows:")
    print("  python3 build_mist_recognition_gt.py --stage label   # GCV -- only real $ cost")
    print("  python3 run_mist_baselines.py --engine tesseract")
    print("  python3 run_mist_baselines.py --engine paddleocr")
    print("  python3 run_mist_baselines.py --engine easyocr")
    print("  python3 run_vlm_baselines.py --engine gpt4v --model gpt-4o   # once funded")
    print("  python3 run_vlm_baselines.py --engine claude --model claude-sonnet-4-5   # once funded")


if __name__ == "__main__":
    main()
