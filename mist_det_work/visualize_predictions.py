#!/usr/bin/env python3
"""Visualize predicted detections vs ground truth on sample MIST test images."""
import argparse
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
TEST_JSON = os.path.expanduser("~/mist/mmocr_annotations/mist_test_textdet.json")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
OUT_DIR = os.path.expanduser("~/mist_det_work/viz_predictions")

GT_COLOR = (0, 255, 0)             # green (BGR)
GT_IGNORE_COLOR = (128, 128, 128)  # gray
PRED_COLOR = (0, 0, 255)           # red


def find_image(raw_path):
    candidates = [
        raw_path,
        os.path.join(IMG_ROOT, os.path.basename(raw_path)),
        os.path.join(os.path.expanduser("~/mist/Images"), raw_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TEST_JSON) as f:
        data_list = json.load(f)["data_list"]

    random.seed(args.seed)
    by_lang = {}
    for item in data_list:
        fname = Path(item["img_path"]).name
        lang = fname.split("__")[0] if "__" in fname else "unknown"
        by_lang.setdefault(lang, []).append(item)

    langs = sorted(by_lang.keys())
    per_lang = max(1, args.num_samples // max(1, len(langs)))
    samples = []
    for lang in langs:
        samples.extend(random.sample(by_lang[lang], min(per_lang, len(by_lang[lang]))))
    if len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)

    print("Loading model (this may take ~10-20s)...")
    from mmocr.apis import TextDetInferencer
    inferencer = TextDetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")

    print(f"Running inference + drawing comparisons on {len(samples)} images -> {OUT_DIR}")
    for item in samples:
        fname = Path(item["img_path"]).name
        img_path = find_image(item["img_path"])
        if img_path is None:
            print(f"  [skip] missing image for {fname}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] failed to read: {img_path}")
            continue

        n_gt, n_gt_ignore = 0, 0
        for inst in item["instances"]:
            poly = np.array(inst["polygon"], dtype=np.int32).reshape(-1, 2)
            ignore = inst.get("ignore", False)
            color = GT_IGNORE_COLOR if ignore else GT_COLOR
            n_gt_ignore += ignore
            n_gt += not ignore
            cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)

        result = inferencer(img_path, return_vis=False)
        pred = result["predictions"][0]
        n_pred = 0
        for poly_flat, score in zip(pred["polygons"], pred["scores"]):
            if score < args.score_thr:
                continue
            n_pred += 1
            poly = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(img, [poly], isClosed=True, color=PRED_COLOR, thickness=2)

        legend = "GREEN=GT  GRAY=GT(ignore)  RED=Pred"
        cv2.putText(img, legend, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, legend, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        out_name = f"{fname.rsplit('.', 1)[0]}_gt{n_gt}_pred{n_pred}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  {fname}: GT={n_gt} (+{n_gt_ignore} ignore), Pred(>={args.score_thr})={n_pred} -> {out_name}")

    print(f"\nDone. Saved to {OUT_DIR} -- open that folder locally to browse.")


if __name__ == "__main__":
    main()
