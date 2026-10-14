#!/usr/bin/env python3
"""
extract_and_route_crops.py
Detect text in sampled MIST test images with the trained DBNet++ checkpoint,
match predictions to real (care=1) ground-truth boxes via polygon IoU, and
route matched crops -- warped to upright rectangles -- into per-language
folders for recognition. Only telugu/hindi/kannada are routed (the only
languages with a trained BSTD recognizer in this project); everything else
(other languages, or predictions with no matching real GT box) is counted
and skipped.

Run from ~/mist_det_work (or anywhere -- all paths below are absolute).

Usage:
    python3 extract_and_route_crops.py --num-images 40
"""
import argparse
import csv
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

CONFIG = os.path.expanduser("~/mist_det_work/mist_dbnetpp_config.py")
CHECKPOINT = os.path.expanduser(
    "~/mist_det_work/work_dirs/mist_dbnetpp/best_mist_hmean_epoch_40.pth"
)
GT_DIR = os.path.expanduser("~/mist/gt/Test")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
OUT_DIR = os.path.expanduser("~/mist_det_work/recognition_crops")

SUPPORTED_LANGUAGES = {"telugu", "hindi", "kannada"}
IOU_MATCH_THRESHOLD = 0.5
CROP_TARGET_HEIGHT = 48


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_crop(img, polygon, target_height=CROP_TARGET_HEIGHT):
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    box = order_points(cv2.boxPoints(rect))
    (tl, tr, br, bl) = box
    width = max(int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))), 1)
    height = max(int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))), 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    scale = target_height / max(height, 1)
    new_w = max(int(width * scale), 1)
    return cv2.resize(warped, (new_w, target_height))


def polygon_iou(poly_a, poly_b):
    """IoU between two arbitrary polygons via local-bbox rasterization
    (no shapely dependency needed)."""
    pa = np.array(poly_a, dtype=np.float32).reshape(-1, 2)
    pb = np.array(poly_b, dtype=np.float32).reshape(-1, 2)
    x_min = int(min(pa[:, 0].min(), pb[:, 0].min())) - 1
    y_min = int(min(pa[:, 1].min(), pb[:, 1].min())) - 1
    x_max = int(max(pa[:, 0].max(), pb[:, 0].max())) + 1
    y_max = int(max(pa[:, 1].max(), pb[:, 1].max())) + 1
    w = max(x_max - x_min, 1)
    h = max(y_max - y_min, 1)
    if w * h > 4_000_000:
        return 0.0
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_a, [(pa - [x_min, y_min]).astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [(pb - [x_min, y_min]).astype(np.int32)], 1)
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=40)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for lang in SUPPORTED_LANGUAGES:
        os.makedirs(os.path.join(OUT_DIR, lang), exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))
    if not gt_files:
        raise SystemExit(f"No GT files found under {GT_DIR}")
    random.seed(args.seed)

    by_lang = {}
    for fp in gt_files:
        stem = Path(fp).stem
        lang = stem.split("__")[0].lower() if "__" in stem else "unknown"
        by_lang.setdefault(lang, []).append(fp)

    langs = sorted(by_lang.keys())
    per_lang = max(1, args.num_images // max(1, len(langs)))
    sampled = []
    for lang in langs:
        sampled.extend(random.sample(by_lang[lang], min(per_lang, len(by_lang[lang]))))
    if len(sampled) > args.num_images:
        sampled = random.sample(sampled, args.num_images)

    print(f"Sampled {len(sampled)} images across {len(langs)} filename-tag buckets.")
    print("Loading detector (this may take ~10-20s)...")
    from mmocr.apis import TextDetInferencer
    inferencer = TextDetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    counts = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    counts["unsupported_or_unmatched"] = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["crop_path", "source_image", "matched_script", "iou", "pred_score"])

        for fp in sampled:
            stem = Path(fp).stem
            img_path = os.path.join(IMG_ROOT, stem + ".png")
            if not os.path.exists(img_path):
                print(f"  [skip] missing image for {stem}")
                continue

            with open(fp, encoding="utf-8") as f:
                gt_data = json.load(f)
            real_shapes = [s for s in gt_data["shapes"] if s.get("care", 0) == 1 and s.get("script")]

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [skip] failed to read {img_path}")
                continue

            result = inferencer(img_path, return_vis=False)
            pred = result["predictions"][0]

            n_this_image = 0
            for poly_flat, score in zip(pred["polygons"], pred["scores"]):
                if score < args.score_thr:
                    continue
                pred_poly = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)

                best_iou, best_script = 0.0, None
                for shape in real_shapes:
                    iou = polygon_iou(pred_poly, shape["points"])
                    if iou > best_iou:
                        best_iou, best_script = iou, shape["script"].lower()

                if best_iou < IOU_MATCH_THRESHOLD or best_script not in SUPPORTED_LANGUAGES:
                    counts["unsupported_or_unmatched"] += 1
                    continue

                crop = warp_crop(img, pred_poly)
                if crop.size == 0 or crop.shape[1] < 4:
                    continue

                crop_name = f"{stem}_{n_this_image}.jpg"
                crop_path = os.path.join(OUT_DIR, best_script, crop_name)
                cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                writer.writerow([os.path.join(best_script, crop_name), stem, best_script,
                                  f"{best_iou:.3f}", f"{score:.3f}"])
                counts[best_script] += 1
                n_this_image += 1

            print(f"  {stem}: {n_this_image} crop(s) routed")

    print("\nDone. Crops routed:")
    for lang in sorted(SUPPORTED_LANGUAGES):
        print(f"  {lang}: {counts[lang]}")
    print(f"  unsupported language / unmatched prediction: {counts['unsupported_or_unmatched']}")
    print(f"\nManifest: {manifest_path}")
    print(f"Crop folders: {[os.path.join(OUT_DIR, l) for l in sorted(SUPPORTED_LANGUAGES)]}")


if __name__ == "__main__":
    main()
