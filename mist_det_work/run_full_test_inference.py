#!/usr/bin/env python3
"""
run_full_test_inference.py
Run the trained DBNet++ checkpoint over the ENTIRE MIST test set (1985
images) and dump raw predicted polygons + scores to JSON -- no extra score
filtering, exactly what the model's own DBPostprocessor emits. This gives
a single, non-cherry-picked operating point for downstream DetEval
scoring (unlike MMOCR's HmeanIOUMetric, which sweeps score thresholds
internally and reports the best one -- see deteval_score.py's docstring).

Run from ~/mist_det_work (or anywhere -- paths below are absolute).

Usage:
    conda activate mist_det
    python3 run_full_test_inference.py
"""
import json
import os
import time
from pathlib import Path

CONFIG = os.path.expanduser("~/mist_det_work/mist_dbnetpp_config.py")
CHECKPOINT = os.path.expanduser(
    "~/mist_det_work/work_dirs/mist_dbnetpp/best_mist_hmean_epoch_40.pth"
)
TEST_JSON = os.path.expanduser("~/mist/mmocr_annotations/mist_test_textdet.json")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
OUT_PATH = os.path.expanduser("~/mist_det_work/full_test_predictions.json")


def main():
    with open(TEST_JSON) as f:
        data_list = json.load(f)["data_list"]

    print(f"Loading detector for {len(data_list)} images...")
    from mmocr.apis import TextDetInferencer
    inferencer = TextDetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")

    predictions = {}
    start = time.time()
    for i, item in enumerate(data_list):
        fname = Path(item["img_path"]).name
        stem = fname.rsplit(".", 1)[0]
        img_path = os.path.join(IMG_ROOT, fname)
        if not os.path.exists(img_path):
            print(f"  [skip] missing image: {fname}")
            continue

        result = inferencer(img_path, return_vis=False)
        pred = result["predictions"][0]

        entries = []
        for poly_flat, score in zip(pred["polygons"], pred["scores"]):
            pts = [[poly_flat[j], poly_flat[j + 1]] for j in range(0, len(poly_flat), 2)]
            entries.append({"points": pts, "score": score})
        predictions[stem] = entries

        if (i + 1) % 100 == 0 or (i + 1) == len(data_list):
            elapsed = time.time() - start
            per_img = elapsed / (i + 1)
            eta = per_img * (len(data_list) - (i + 1))
            print(f"  [{i + 1}/{len(data_list)}] {per_img:.2f}s/img, ETA {eta / 60:.1f} min")

    with open(OUT_PATH, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved predictions for {len(predictions)} images to {OUT_PATH}")


if __name__ == "__main__":
    main()
