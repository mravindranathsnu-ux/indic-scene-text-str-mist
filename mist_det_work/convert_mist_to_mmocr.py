#!/usr/bin/env python3
"""Convert MIST's LabelMe-style ground truth into MMOCR's unified textdet JSON format."""
import json
import os
import sys
from pathlib import Path
from PIL import Image

MIST_ROOT = Path.home() / "mist"
OUT_DIR = MIST_ROOT / "mmocr_annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "Train",
    "test": "Test",
}

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]

def find_image(img_dir, jf, ann):
    """Try both naming conventions seen across MIST splits: the JSON's own
    filename (Test images are renamed with a language prefix to match this),
    and the raw imagePath basename (Train images keep this unprefixed name)."""
    candidates = []
    raw = ann.get("imagePath")
    if raw:
        candidates.append(os.path.basename(raw.replace("\\", "/")))
    candidates.append(jf.stem + ".png")
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for c in candidates:
        p = img_dir / c
        if p.is_file():
            return p, c
    return None, candidates

def convert_split(split_name, dir_name):
    img_dir = MIST_ROOT / "Images" / dir_name
    gt_dir = MIST_ROOT / "gt" / dir_name

    if not img_dir.is_dir() or not gt_dir.is_dir():
        print(f"[{split_name}] missing directory: {img_dir} or {gt_dir}", file=sys.stderr)
        return

    json_files = sorted(gt_dir.glob("*.json"))
    data_list = []
    stats = {"images": 0, "images_missing": 0, "images_unreadable": 0,
              "instances": 0, "ignored_instances": 0, "degenerate_skipped": 0,
              "json_errors": 0}
    bad_files = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            stats["json_errors"] += 1
            bad_files.append(("json_error", str(jf), str(e)))
            continue

        img_path, image_name_or_candidates = find_image(img_dir, jf, ann)
        if img_path is None:
            stats["images_missing"] += 1
            bad_files.append(("missing", str(jf), f"tried: {image_name_or_candidates}"))
            continue
        image_name = image_name_or_candidates

        width = ann.get("imageWidth")
        height = ann.get("imageHeight")
        if not width or not height:
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                stats["images_unreadable"] += 1
                bad_files.append(("unreadable", str(img_path), str(e)))
                continue

        instances = []
        for shape in ann.get("shapes", []):
            points = shape.get("points", [])
            if len(points) < 3:
                stats["degenerate_skipped"] += 1
                continue

            polygon = [float(c) for pt in points for c in pt]
            bbox = polygon_to_bbox(points)
            care = shape.get("care", 1)
            ignore = (care != 1)

            instances.append({
                "polygon": polygon,
                "bbox": bbox,
                "bbox_label": 0,
                "ignore": ignore,
            })
            stats["instances"] += 1
            if ignore:
                stats["ignored_instances"] += 1

        data_list.append({
            "img_path": image_name,
            "height": height,
            "width": width,
            "instances": instances,
        })
        stats["images"] += 1

    out = {
        "metainfo": {
            "dataset_type": "TextDetDataset",
            "task_name": "textdet",
            "category": [{"id": 0, "name": "text"}],
        },
        "data_list": data_list,
    }

    out_path = OUT_DIR / f"mist_{split_name}_textdet.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"[{split_name}] wrote {out_path}")
    print(f"[{split_name}] stats: {stats}")
    if bad_files:
        bad_path = OUT_DIR / f"mist_{split_name}_bad_files.json"
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_files, f, indent=2)
        print(f"[{split_name}] logged {len(bad_files)} problem file(s) to {bad_path}")


for split_name, dir_name in SPLITS.items():
    convert_split(split_name, dir_name)
