#!/usr/bin/env python3
"""
tiled_inference_sample.py
SAHI-style tiled inference experiment: does splitting each image into
overlapping tiles (each fed to the SAME trained checkpoint) recover
small/dense text that whole-image inference misses -- with NO retraining?

Native MIST test images are 1920x1080. The existing test_pipeline resizes
to ~1820x1024 (keep_ratio, scale=(4068,1024)) -- only a ~5% shrink, so the
recall gap is not primarily a test-time downsizing problem. It's much more
likely a TRAINING-time one: train_pipeline resizes crops down to 640x640,
roughly a 3x shrink from native, so the model has mostly learned to
recognize text at that heavily-downsized apparent scale.

Tiling doesn't fix training, but it changes what the model SEES at test
time: a 640x640 tile of a 1920x1080 frame shows text much closer to its
native pixel size than a whole-frame pass does -- and the existing
test_pipeline will even upscale that tile ~1.6x further (640 -> ~1024),
since it's well under the (4068,1024) cap. This is exactly the situation
where tiled/sliding-window inference is known to help small-object
recall (the "SAHI" technique), without retraining.

Run on a modest stratified SAMPLE first (not all 1985 images), and compare
against the SAME sample's existing whole-image predictions -- an
apples-to-apples subset comparison in minutes, before committing to a
full-test-set rescore.

Usage:
    conda activate mist_det
    python3 tiled_inference_sample.py --num-images 300
    python3 deteval_score.py --tr 0.8 --tp 0.4 --pred tiled_sample_baseline.json
    python3 deteval_score.py --tr 0.8 --tp 0.4 --pred tiled_sample_tiled.json
    python3 deteval_score.py --tr 0.7 --tp 0.6 --pred tiled_sample_baseline.json
    python3 deteval_score.py --tr 0.7 --tp 0.6 --pred tiled_sample_tiled.json
"""
import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

CONFIG = os.path.expanduser("~/mist_det_work/mist_dbnetpp_config.py")
CHECKPOINT = os.path.expanduser(
    "~/mist_det_work/work_dirs/mist_dbnetpp/best_mist_hmean_epoch_40.pth"
)
GT_DIR = os.path.expanduser("~/mist/gt/Test")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
FULL_PREDICTIONS = os.path.expanduser("~/mist_det_work/full_test_predictions.json")
TILE_TMP_DIR = os.path.expanduser("~/mist_det_work/.tile_tmp")

TILE_SIZE = 640
OVERLAP = 160
NMS_IOU_THR = 0.3


def polygon_iou(poly_a, poly_b):
    """Local-bbox-rasterized polygon IoU (no shapely dependency) -- same
    approach already used and proven in extract_and_route_crops.py."""
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


def make_tiles(w, h, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Overlapping tile origins covering (w, h). Every tile is a real,
    full-size crop of the image (edge tiles shift inward rather than
    getting padded), so no tile is smaller than tile_size unless the
    image itself is smaller than tile_size."""
    step = max(tile_size - overlap, 1)
    xs = list(range(0, max(w - tile_size, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile_size, 0) + 1, step)) or [0]
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    tiles = []
    for y in ys:
        for x in xs:
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
            tiles.append((x0, y0, x1, y1))
    return tiles


def nms_polygons(entries, iou_thr=NMS_IOU_THR):
    """Greedy polygon NMS: sort by score desc, keep a box unless it
    overlaps (IoU > iou_thr) a higher-scored box already kept. This is
    what removes the double-detections tile overlap regions produce."""
    entries = sorted(entries, key=lambda e: e["score"], reverse=True)
    kept = []
    for e in entries:
        if all(polygon_iou(e["points"], k["points"]) <= iou_thr for k in kept):
            kept.append(e)
    return kept


def sample_stems(num_images, seed=42):
    """Same stratified-by-language-tag sampling as extract_and_route_crops.py,
    for a representative (not just easy) sample."""
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))
    by_lang = {}
    for fp in gt_files:
        stem = Path(fp).stem
        lang = stem.split("__")[0].lower() if "__" in stem else "unknown"
        by_lang.setdefault(lang, []).append(stem)
    random.seed(seed)
    langs = sorted(by_lang.keys())
    per_lang = max(1, num_images // max(1, len(langs)))
    sampled = []
    for lang in langs:
        sampled.extend(random.sample(by_lang[lang], min(per_lang, len(by_lang[lang]))))
    if len(sampled) > num_images:
        sampled = random.sample(sampled, num_images)
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--tile-size", type=int, default=TILE_SIZE)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out-prefix", type=str, default="tiled_sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stems = sample_stems(args.num_images, seed=args.seed)
    print(f"Sampled {len(stems)} images.")

    # Baseline subset: pulled straight from the already-computed whole-image
    # epoch-40 predictions, no re-inference needed -- guarantees a fair,
    # identical-images comparison against the tiled result below.
    with open(FULL_PREDICTIONS) as f:
        full_preds = json.load(f)
    baseline = {s: full_preds[s] for s in stems if s in full_preds}
    baseline_path = f"{args.out_prefix}_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f)
    print(f"Baseline (whole-image) subset written to {baseline_path} "
          f"({len(baseline)}/{len(stems)} found).")

    os.makedirs(TILE_TMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TILE_TMP_DIR, "tile.png")

    print("Loading detector for tiled inference...")
    from mmocr.apis import TextDetInferencer
    inferencer = TextDetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")

    tiled_preds = {}
    start = time.time()
    for i, stem in enumerate(stems):
        img_path = os.path.join(IMG_ROOT, stem + ".png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] failed to read {img_path}")
            continue
        h, w = img.shape[:2]
        tiles = make_tiles(w, h, args.tile_size, args.overlap)

        all_entries = []
        for (x0, y0, x1, y1) in tiles:
            crop = img[y0:y1, x0:x1]
            cv2.imwrite(tmp_path, crop)
            result = inferencer(tmp_path, return_vis=False)
            pred = result["predictions"][0]
            for poly_flat, score in zip(pred["polygons"], pred["scores"]):
                if score < args.score_thr:
                    continue
                pts = [[poly_flat[j] + x0, poly_flat[j + 1] + y0]
                       for j in range(0, len(poly_flat), 2)]
                all_entries.append({"points": pts, "score": score})

        merged = nms_polygons(all_entries)
        tiled_preds[stem] = merged

        if (i + 1) % 25 == 0 or (i + 1) == len(stems):
            elapsed = time.time() - start
            per_img = elapsed / (i + 1)
            eta = per_img * (len(stems) - (i + 1))
            print(f"  [{i + 1}/{len(stems)}] {per_img:.2f}s/img, "
                  f"ETA {eta / 60:.1f} min, {len(tiles)} tiles/img, "
                  f"{len(merged)} boxes (last image)")

    tiled_path = f"{args.out_prefix}_tiled.json"
    with open(tiled_path, "w") as f:
        json.dump(tiled_preds, f)
    print(f"\nTiled predictions written to {tiled_path}")
    print("\nNow score both subsets head-to-head:")
    print(f"  python3 deteval_score.py --tr 0.8 --tp 0.4 --pred {baseline_path}")
    print(f"  python3 deteval_score.py --tr 0.8 --tp 0.4 --pred {tiled_path}")
    print(f"  python3 deteval_score.py --tr 0.7 --tp 0.6 --pred {baseline_path}")
    print(f"  python3 deteval_score.py --tr 0.7 --tp 0.6 --pred {tiled_path}")


if __name__ == "__main__":
    main()
