#!/usr/bin/env python3
"""
build_mist_recognition_gt.py

Builds a pseudo-ground-truth recognition eval set for MIST, since the
released MIST annotations (~/mist/gt/{Train,Test}/*.json) carry only
{points, care, script} -- no transcription field. Direct inspection of
the actual downloaded data confirmed this; the earlier ICVGIP26 paper's
Future Work claim that MIST "provides ... transcription" labels was
written without carefully checking the released data and should not be
trusted over this.

Approach (per user's decision, Sep 18 2026):
  1. Use MIST's own REAL ground-truth polygons (not our detector's
     predictions) to extract per-instance crops from the Test split.
     This decouples recognition scoring from our detector's imperfect
     recall -- we're evaluating recognition, not detection+recognition
     end-to-end, so the crop boundaries should be trustworthy.
  2. Stratified-sample a fixed number of instances per language (default
     300), seeded for reproducibility (same seed=42 convention used
     earlier this session for tiled_inference_sample.py).
  3. Run each crop through Google Cloud Vision's TEXT_DETECTION, and
     treat the returned text as the pseudo-ground-truth transcription
     for that instance. Google Cloud Vision was chosen because it's the
     same OCR system MIST's own authors used for their initial
     detection-annotation pass (per the paper), so pseudo-labels here
     are at least methodologically consistent with how the dataset
     itself was built -- NOT because it's assumed perfect. This is a
     noisy pseudo-GT, not human-verified ground truth, and the eventual
     paper needs to say so explicitly.

Two auth paths are supported, so the actual API key/credentials never
have to pass through this conversation -- set ONE of these as an
environment variable in your own shell before running:
  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
        (uses the google-cloud-vision client library)
  - GCV_API_KEY=your-api-key
        (uses the plain REST endpoint, no extra client library needed
        beyond `requests`)

Output:
  ~/mist_det_work/recognition_gt/<language>/<instance_id>.png   (crops)
  ~/mist_det_work/recognition_gt/manifest.csv
      columns: instance_id, language, image_stem, crop_path,
               points (json), gcv_text, gcv_confidence, status

Usage:
  conda activate mist_det
  cd ~/mist_det_work
  # step 1: extract crops + sample (no network / no API calls yet)
  python3 build_mist_recognition_gt.py --stage extract --per-language 300
  # step 2: label the sampled crops via Google Cloud Vision
  python3 build_mist_recognition_gt.py --stage label
  # (both stages can also be run together with --stage all)
"""
import argparse
import base64
import csv
import glob
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

GT_DIR = os.path.expanduser("~/mist/gt/Test")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
OUT_ROOT = os.path.expanduser("~/mist_det_work/recognition_gt")
MANIFEST_PATH = os.path.join(OUT_ROOT, "manifest.csv")

# The 10 languages Phase 1 (multilingual-str-clip) covers, per the
# ICVGIP26 paper's Table 2. MIST's own test-split language breakdown
# (confirmed earlier this session) uses these exact 10 -- no English
# instances were found in the test split despite the paper listing
# English as an 11th language.
TARGET_LANGUAGES = {
    "kannada", "malayalam", "marathi", "bengali", "telugu",
    "punjabi", "hindi", "odia", "oriya", "tamil", "gujarati",
}
# Normalize "oriya" (MIST's own tag, seen in some releases) to "odia"
# (Phase 1's naming) so the two line up as one language downstream.
LANG_NORMALIZE = {"oriya": "odia"}

PAD = 2  # small pixel padding around each polygon's bounding box


def resolve_image_path(stem):
    """Test images were renamed at release time to carry a language
    prefix that the GT json's internal imagePath field doesn't reflect
    (a quirk already worked around in convert_mist_to_mmocr.py). Try
    the json's own stem first (this already matches on-disk Test
    filenames), then fall back to the raw imagePath basename."""
    candidate = os.path.join(IMG_ROOT, stem + ".png")
    if os.path.exists(candidate):
        return candidate
    return None


def load_instances():
    """Walk every Test GT json, keep care==1 instances whose script is
    one of the 10 target languages, normalize language spelling, and
    return them grouped by language."""
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))
    by_lang = {}
    skipped_no_image = 0
    for fp in gt_files:
        stem = Path(fp).stem
        img_path = resolve_image_path(stem)
        if img_path is None:
            skipped_no_image += 1
            continue
        with open(fp) as f:
            data = json.load(f)
        for idx, shape in enumerate(data.get("shapes", [])):
            if shape.get("care") != 1:
                continue
            lang = (shape.get("script") or "").strip().lower()
            lang = LANG_NORMALIZE.get(lang, lang)
            if lang not in TARGET_LANGUAGES or lang == "oriya":
                continue
            by_lang.setdefault(lang, []).append({
                "image_stem": stem,
                "image_path": img_path,
                "shape_idx": idx,
                "points": shape["points"],
            })
    if skipped_no_image:
        print(f"  [warn] {skipped_no_image} GT files had no matching "
              f"on-disk image, skipped.")
    return by_lang


def crop_instance(img, points, pad=PAD):
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    h, w = img.shape[:2]
    x0 = max(int(pts[:, 0].min()) - pad, 0)
    y0 = max(int(pts[:, 1].min()) - pad, 0)
    x1 = min(int(pts[:, 0].max()) + pad, w)
    y1 = min(int(pts[:, 1].max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def stage_extract(per_language, seed):
    by_lang = load_instances()
    print("Instances available per language (care=1, target languages):")
    for lang in sorted(by_lang):
        print(f"  {lang:10s} {len(by_lang[lang])}")

    random.seed(seed)
    rows = []
    os.makedirs(OUT_ROOT, exist_ok=True)
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        k = min(per_language, len(pool))
        sample = random.sample(pool, k)
        lang_dir = os.path.join(OUT_ROOT, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # Cache loaded images across instances from the same image.
        img_cache = {}
        n_written = 0
        for i, inst in enumerate(sample):
            img = img_cache.get(inst["image_path"])
            if img is None:
                img = cv2.imread(inst["image_path"])
                img_cache[inst["image_path"]] = img
            if img is None:
                continue
            crop = crop_instance(img, inst["points"])
            if crop is None or crop.size == 0:
                continue
            instance_id = f"{lang}_{i:04d}_{inst['image_stem']}_{inst['shape_idx']}"
            crop_path = os.path.join(lang_dir, instance_id + ".png")
            cv2.imwrite(crop_path, crop)
            rows.append({
                "instance_id": instance_id,
                "language": lang,
                "image_stem": inst["image_stem"],
                "crop_path": crop_path,
                "points": json.dumps(inst["points"]),
                "gcv_text": "",
                "gcv_confidence": "",
                "status": "pending",
            })
            n_written += 1
        print(f"  {lang:10s} sampled {k}, wrote {n_written} crops")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_id", "language", "image_stem", "crop_path",
            "points", "gcv_text", "gcv_confidence", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written: {MANIFEST_PATH} ({len(rows)} rows)")
    print("Next: python3 build_mist_recognition_gt.py --stage label")


def gcv_detect_text_apikey(image_path, api_key, session):
    with open(image_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
        }]
    }
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    resp = session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()["responses"][0]
    if "error" in result:
        return None, None, f"error:{result['error'].get('message', 'unknown')}"
    annotations = result.get("textAnnotations", [])
    if not annotations:
        return "", None, "no_text_detected"
    return annotations[0].get("description", "").strip(), None, "ok"


def gcv_detect_text_client(image_path, client, vision):
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        return None, None, f"error:{response.error.message}"
    annotations = response.text_annotations
    if not annotations:
        return "", None, "no_text_detected"
    return annotations[0].description.strip(), None, "ok"


def _checkpoint(rows):
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, MANIFEST_PATH)


def stage_label(rate_limit_sleep):
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"No manifest at {MANIFEST_PATH} -- run --stage extract first.")

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    api_key = os.environ.get("GCV_API_KEY")
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    use_client, client, vision = False, None, None
    session = None
    if cred_path:
        try:
            from google.cloud import vision as vision_mod
            client = vision_mod.ImageAnnotatorClient()
            vision = vision_mod
            use_client = True
            print("Using google-cloud-vision client library "
                  f"(GOOGLE_APPLICATION_CREDENTIALS={cred_path}).")
        except ImportError:
            raise SystemExit(
                "GOOGLE_APPLICATION_CREDENTIALS is set but google-cloud-vision "
                "isn't installed. Run: pip install google-cloud-vision")
    elif api_key:
        import requests
        session = requests.Session()
        print("Using Google Cloud Vision REST API with GCV_API_KEY.")
    else:
        raise SystemExit(
            "Set either GOOGLE_APPLICATION_CREDENTIALS (service-account json path) "
            "or GCV_API_KEY (API key) as an environment variable first.")

    n_ok, n_err, n_empty = 0, 0, 0
    start = time.time()
    for i, row in enumerate(rows):
        if row["status"] in ("ok", "no_text_detected"):
            # Resumable: skip already-labeled rows. Both are valid,
            # final results -- "no_text_detected" isn't a failure to
            # retry, it's Google Vision's genuine answer for that crop.
            # (Previously only "ok" was skipped here, unlike every other
            # script in this project, so a re-run silently re-billed
            # every previously-empty crop too.)
            continue
        try:
            if use_client:
                text, conf, status = gcv_detect_text_client(row["crop_path"], client, vision)
            else:
                text, conf, status = gcv_detect_text_apikey(row["crop_path"], api_key, session)
        except Exception as e:
            text, conf, status = None, None, f"exception:{e}"

        row["gcv_text"] = text if text is not None else ""
        row["gcv_confidence"] = "" if conf is None else conf
        row["status"] = status
        if status == "ok":
            n_ok += 1
        elif status == "no_text_detected":
            n_empty += 1
        else:
            n_err += 1

        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - start
            print(f"  [{i + 1}/{len(rows)}] ok={n_ok} empty={n_empty} "
                  f"err={n_err} ({elapsed:.0f}s elapsed)")
            # Checkpoint the manifest periodically so a crash/rate-limit
            # doesn't lose already-labeled work. Atomic + encoding-safe,
            # same as run_mist_baselines.py / run_vlm_baselines.py: write
            # to a sibling .tmp file, then os.replace() it over the real
            # manifest, so a mid-write crash can never truncate it.
            _checkpoint(rows)

    _checkpoint(rows)
    print(f"\nDone. ok={n_ok} empty={n_empty} err={n_err}. "
          f"Manifest updated: {MANIFEST_PATH}")
    if n_err:
        print("Re-run the same command to retry only the failed/pending rows "
              "(already-'ok' rows are skipped).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "label", "all"], default="all")
    ap.add_argument("--per-language", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rate-limit-sleep", type=float, default=0.0,
                     help="Seconds to sleep between API calls, if you hit quota limits.")
    args = ap.parse_args()

    if args.stage in ("extract", "all"):
        stage_extract(args.per_language, args.seed)
    if args.stage in ("label", "all"):
        stage_label(args.rate_limit_sleep)


if __name__ == "__main__":
    main()
