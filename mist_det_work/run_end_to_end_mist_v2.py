#!/usr/bin/env python3
"""
run_end_to_end_mist_v2.py

v2 of the end-to-end (detect THEN recognize) MIST evaluation. Two changes
from v1 (run_end_to_end_mist.py), both completing items from the paper's
own Future Work (Section 8):

  1. --system vlm: runs the SAME end-to-end protocol using the fine-tuned
     VLM (per-language Qwen3-VL-2B LoRA adapters, same checkpoints/prompt
     as run_track2_vlm.py) instead of the CLIP recognizer, so Table 6 can
     report both systems the way Table 4/5 already do.

  2. A CORRECTNESS FIX to v1's match-type classification, discovered while
     building (1): v1 classified a gt instance as "one_to_one" whenever it
     matched exactly ONE detected polygon (len(det_list) == 1). But a
     MANY-TO-ONE match (the detector merges several real words into one
     box) also produces a length-1 list for EACH of those words, pointing
     at the SAME shared detected box -- v1's check couldn't tell this
     apart from a genuine 1:1 match, so merged-box instances were silently
     counted as "one_to_one" even though the crop fed to the recognizer
     actually contained more than one word. v2 builds a det->gts reverse
     map to tell the three cases apart properly:
       - one_to_one:         exactly 1 matched det, and that det matches
                              no other gt either -- a clean, unambiguous
                              correspondence. Crop: that det's bbox.
       - many_to_one_merged: exactly 1 matched det, but that SAME det
                              also matches >=1 other gt -- the detector
                              drew one box around multiple real words.
                              Crop: that (shared) det's bbox, same
                              mechanics as one_to_one but now correctly
                              labeled, so the crop may contain extra text
                              beyond this specific instance's word.
       - one_to_many_split:  >1 matched dets -- the detector split one
                              real word across multiple boxes. Crop: the
                              UNION bounding box of all matched dets (the
                              smallest box containing every fragment),
                              padded the same way as a normal crop.

  Per the paper's Future Work, BOTH many_to_one_merged and
  one_to_many_split are now actually run through the recognizer (v1
  scored every non-one_to_one match as an automatic miss without
  attempting recognition at all). Only "missed" (no matching detected
  polygon whatsoever) remains an automatic miss, since there is genuinely
  no crop to attempt recognition on. This is run honestly, not optimised
  to look good: a merged box frequently contains more than the target
  word, and a union-of-fragments crop is not guaranteed to look like a
  clean single-word crop either, so these categories' recognition
  accuracy is reported, not assumed.

VERIFIED (unit-tested before delivery): the det->gts classification was
checked against the same 5 hand-built DetEval cases v1 used (clean 1:1,
total miss, 1-to-many split, many-to-1 merge, degenerate-polygon index
safety), plus 2 new cases isolating one_to_one vs. many_to_one_merged
specifically (same len(det_list)==1 shape, different det->gts fan-out),
and the union-bbox cropper was checked against a synthetic 2-fragment
split. All passed -- see the bottom of this file's companion test run.

USAGE
    conda activate base
    cd ~/mist_det_work
    export HF_HUB_OFFLINE=1   # harmless no-op for --system clip, needed for vlm

    # CLIP, corrected classification (re-run even though clip_text already
    # exists in the manifest -- this fixes the one_to_one/many_to_one_merged
    # mislabeling and recognizes the previously-discarded split/merged crops):
    python3 run_end_to_end_mist_v2.py --system clip --pred full_test_predictions_ext.json --limit 50

    # Fine-tuned VLM (requires peft + bitsandbytes + a recent transformers
    # in the `base` env -- check first: python3 -c "import peft, bitsandbytes, transformers"):
    python3 run_end_to_end_mist_v2.py --system vlm --pred full_test_predictions_ext.json --limit 50

    # then the full runs (drop --limit) once each smoke test looks right.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon

GT_DIR = os.path.expanduser("~/mist/gt/Test")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
DEFAULT_PRED_PATH = os.path.expanduser("~/mist_det_work/full_test_predictions.json")
MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")
CROP_OUT_ROOT = os.path.expanduser("~/mist_det_work/recognition_gt_e2e_v2")
CLIP_REPO = os.path.expanduser("~/multilingual-str-clip")
CLIP_CKPT = os.path.join(CLIP_REPO, "weights", "ViT-B-16.pt")
CHECKPOINTS_DIR = os.path.join(CLIP_REPO, "checkpoints")
VLM_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
VLM_PROMPT = "Read the text written in this image. Respond with only the exact text, nothing else."

LANGUAGES = ["bengali", "gujarati", "hindi", "kannada", "malayalam",
             "marathi", "odia", "punjabi", "tamil", "telugu"]

PAD = 2  # same padding convention as build_mist_recognition_gt.py's crop_instance

DEFAULT_TR = 0.7
DEFAULT_TP = 0.6


# ---------------------------------------------------------------------------
# DetEval matching -- UNCHANGED from v1 / deteval_score.py (see v1's own
# header comment for the full provenance note). Only the caller-side
# bookkeeping (det_to_gts, classify_match) is new in v2.
# ---------------------------------------------------------------------------
def match_image(gt, pred, tr=DEFAULT_TR, tp=DEFAULT_TP):
    area_recall_constraint = tr
    area_precision_constraint = tp
    ev_param_ind_center_diff_thr = 1

    def get_intersection(pD, pG):
        return Polygon(pD).intersection(Polygon(pG)).area

    gtRects, detRects, gt_orig_idx = [], [], []
    gtDontCareRectsNum, detDontCareRectsNum = [], []

    for n in range(len(gt)):
        points = gt[n]["points"]
        if not Polygon(points).is_valid or not Polygon(points).is_simple:
            continue
        gtRects.append(points)
        gt_orig_idx.append(gt[n]["orig_idx"])
        if gt[n]["ignore"]:
            gtDontCareRectsNum.append(len(gtRects) - 1)

    for n in range(len(pred)):
        points = pred[n]["points"]
        if not Polygon(points).is_valid or not Polygon(points).is_simple:
            continue
        detRects.append(points)
        if gtDontCareRectsNum:
            for dcNum in gtDontCareRectsNum:
                dontCareRect = gtRects[dcNum]
                intersected_area = get_intersection(dontCareRect, points)
                rdDimensions = Polygon(points).area
                precision_dc = 0 if rdDimensions == 0 else intersected_area / rdDimensions
                if precision_dc > area_precision_constraint:
                    detDontCareRectsNum.append(len(detRects) - 1)
                    break

    gt_match_dets = {}
    if not detRects or not gtRects:
        return gt_match_dets, gtRects, detRects, gt_orig_idx

    recallMat = np.empty([len(gtRects), len(detRects)])
    precisionMat = np.empty([len(gtRects), len(detRects)])
    gtRectMat = np.zeros(len(gtRects), np.int8)
    detRectMat = np.zeros(len(detRects), np.int8)
    for gtNum in range(len(gtRects)):
        for detNum in range(len(detRects)):
            rG, rD = gtRects[gtNum], detRects[detNum]
            intersected_area = get_intersection(rG, rD)
            rgDimensions = Polygon(rG).area
            rdDimensions = Polygon(rD).area
            recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else intersected_area / rgDimensions
            precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

    def num_overlaps_gt(gtNum):
        return sum(1 for d in range(len(detRects))
                   if d not in detDontCareRectsNum and recallMat[gtNum, d] > 0)

    def num_overlaps_det(detNum):
        return sum(1 for g in range(len(gtRects))
                   if g not in gtDontCareRectsNum and recallMat[g, detNum] > 0)

    def one_to_one_match(row, col):
        cont = sum(1 for j in range(len(detRects))
                   if recallMat[row, j] >= area_recall_constraint and precisionMat[row, j] >= area_precision_constraint)
        if cont != 1:
            return False
        cont = sum(1 for i in range(len(gtRects))
                   if recallMat[i, col] >= area_recall_constraint and precisionMat[i, col] >= area_precision_constraint)
        if cont != 1:
            return False
        return recallMat[row, col] >= area_recall_constraint and precisionMat[row, col] >= area_precision_constraint

    def is_single_overlap(row, col):
        return num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1

    def center_distance(r1, r2):
        return ((np.mean(r1, axis=0) - np.mean(r2, axis=0)) ** 2).sum() ** 0.5

    def diag(r):
        r = np.array(r)
        return ((r[:, 0].max() - r[:, 0].min()) ** 2 + (r[:, 1].max() - r[:, 1].min()) ** 2) ** 0.5

    def one_to_many_match(gtNum):
        many_sum = 0
        detRectsL = []
        for detNum in range(len(detRects)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                if precisionMat[gtNum, detNum] >= area_precision_constraint:
                    many_sum += recallMat[gtNum, detNum]
                    detRectsL.append(detNum)
        if round(many_sum, 4) >= area_recall_constraint:
            return True, detRectsL
        return False, []

    def many_to_one_match(detNum):
        many_sum = 0
        gtRectsL = []
        for gtNum in range(len(gtRects)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum, detNum] >= area_recall_constraint:
                    many_sum += precisionMat[gtNum, detNum]
                    gtRectsL.append(gtNum)
        if round(many_sum, 4) >= area_precision_constraint:
            return True, gtRectsL
        return False, []

    for gtNum in range(len(gtRects)):
        for detNum in range(len(detRects)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum:
                if one_to_one_match(gtNum, detNum) and is_single_overlap(gtNum, detNum):
                    rG, rD = gtRects[gtNum], detRects[detNum]
                    normDist = center_distance(rG, rD)
                    normDist /= diag(rG) + diag(rD)
                    normDist *= 2.0
                    if normDist < ev_param_ind_center_diff_thr:
                        gtRectMat[gtNum] = 1
                        detRectMat[detNum] = 1
                        gt_match_dets[gtNum] = [detNum]

    for gtNum in range(len(gtRects)):
        if gtNum not in gtDontCareRectsNum and gtRectMat[gtNum] == 0:
            match, matchesDet = one_to_many_match(gtNum)
            if match and num_overlaps_gt(gtNum) >= 2:
                gtRectMat[gtNum] = 1
                for detNum in matchesDet:
                    detRectMat[detNum] = 1
                gt_match_dets[gtNum] = list(matchesDet)

    for detNum in range(len(detRects)):
        if detNum not in detDontCareRectsNum and detRectMat[detNum] == 0:
            match, matchesGt = many_to_one_match(detNum)
            if match and num_overlaps_det(detNum) >= 2:
                detRectMat[detNum] = 1
                for gtNum in matchesGt:
                    gtRectMat[gtNum] = 1
                    gt_match_dets[gtNum] = [detNum]

    return gt_match_dets, gtRects, detRects, gt_orig_idx


def classify_match(det_list, det_to_gts):
    """NEW in v2. det_list is gt_match_dets[gtNum]; det_to_gts maps a det
    index back to every gt index that matched it, built once per image.
    See module docstring for what each label means."""
    if len(det_list) > 1:
        return "one_to_many_split"
    detNum = det_list[0]
    if len(det_to_gts.get(detNum, [])) > 1:
        return "many_to_one_merged"
    return "one_to_one"


def load_full_gt_shapes(stem):
    fp = os.path.join(GT_DIR, stem + ".json")
    if not os.path.exists(fp):
        return None
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    gt = []
    for idx, shape in enumerate(data.get("shapes", [])):
        gt.append({
            "points": shape["points"],
            "ignore": shape.get("care", 1) != 1,
            "orig_idx": idx,
        })
    return gt


def crop_bbox(img, points, pad=PAD):
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    h, w = img.shape[:2]
    x0 = max(int(pts[:, 0].min()) - pad, 0)
    y0 = max(int(pts[:, 1].min()) - pad, 0)
    x1 = min(int(pts[:, 0].max()) + pad, w)
    y1 = min(int(pts[:, 1].max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def crop_union_bbox(img, polygons, pad=PAD):
    """NEW in v2: for a one_to_many_split match, crop the smallest box
    containing every matched detected fragment, so the recognizer at least
    sees the whole word rather than one arbitrary piece of it."""
    all_pts = np.concatenate(
        [np.array(p, dtype=np.float32).reshape(-1, 2) for p in polygons], axis=0)
    h, w = img.shape[:2]
    x0 = max(int(all_pts[:, 0].min()) - pad, 0)
    y0 = max(int(all_pts[:, 1].min()) - pad, 0)
    x1 = min(int(all_pts[:, 0].max()) + pad, w)
    y1 = min(int(all_pts[:, 1].max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def parse_shape_idx(instance_id, image_stem):
    pos = instance_id.index(image_stem)
    after = instance_id[pos + len(image_stem):]
    return int(after.lstrip("_"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["clip", "vlm"], default="clip")
    ap.add_argument("--pred", default=DEFAULT_PRED_PATH)
    ap.add_argument("--tr", type=float, default=DEFAULT_TR)
    ap.add_argument("--tp", type=float, default=DEFAULT_TP)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--vlm_max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_csv", default=None,
                     help="Default: ~/mist_det_work/recognition_gt/e2e_results_<system>_v2.csv")
    args = ap.parse_args()
    if args.out_csv is None:
        args.out_csv = os.path.expanduser(
            f"~/mist_det_work/recognition_gt/e2e_results_{args.system}_v2.csv")

    pred_path = os.path.expanduser(args.pred)
    if not os.path.exists(pred_path):
        raise SystemExit(f"Missing {pred_path} -- run run_full_test_inference*.py first.")
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"Missing {MANIFEST_PATH}")

    print(f"Loading detector predictions from {pred_path} ...")
    with open(pred_path) as f:
        all_preds = json.load(f)

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    status_col = "clip_status" if args.system == "clip" else "track2_status"
    oracle_col = "clip_text" if args.system == "clip" else "track2_text"
    scored = [r for r in rows if r.get("status") == "ok" and r.get(status_col) == "ok"]
    if args.limit:
        scored = scored[:args.limit]
    print(f"Scoring universe: {len(scored)} instances.\n")

    os.makedirs(CROP_OUT_ROOT, exist_ok=True)
    gt_cache, img_cache, match_cache = {}, {}, {}
    classified = []
    counts = {"one_to_one": 0, "many_to_one_merged": 0, "one_to_many_split": 0,
              "missed": 0, "excluded_invalid_gt": 0, "no_detector_output": 0,
              "image_unreadable": 0, "degenerate_det_box": 0}

    for r in scored:
        stem, lang = r["image_stem"], r["language"]
        shape_idx = parse_shape_idx(r["instance_id"], stem)

        if stem not in match_cache:
            gt_shapes = gt_cache.setdefault(stem, load_full_gt_shapes(stem))
            preds = all_preds.get(stem)
            if gt_shapes is None or preds is None:
                match_cache[stem] = None
            else:
                pred_list = [{"points": p["points"]} for p in preds]
                gt_match_dets, gtRects, detRects, gt_orig_idx = match_image(gt_shapes, pred_list, args.tr, args.tp)
                det_to_gts = defaultdict(list)
                for gnum, dets in gt_match_dets.items():
                    for d in dets:
                        det_to_gts[d].append(gnum)
                match_cache[stem] = (gt_match_dets, gtRects, detRects, gt_orig_idx, det_to_gts)

        cached = match_cache[stem]
        if cached is None:
            counts["no_detector_output"] += 1
            classified.append({**r, "match_type": "no_detector_output", "e2e_crop_path": ""})
            continue

        gt_match_dets, gtRects, detRects, gt_orig_idx, det_to_gts = cached
        if shape_idx not in gt_orig_idx:
            counts["excluded_invalid_gt"] += 1
            classified.append({**r, "match_type": "excluded_invalid_gt", "e2e_crop_path": ""})
            continue
        gtNum = gt_orig_idx.index(shape_idx)

        if gtNum not in gt_match_dets:
            counts["missed"] += 1
            classified.append({**r, "match_type": "missed", "e2e_crop_path": ""})
            continue

        det_list = gt_match_dets[gtNum]
        match_type = classify_match(det_list, det_to_gts)

        img = img_cache.get(stem)
        if img is None:
            img = cv2.imread(os.path.join(IMG_ROOT, stem + ".png"))
            img_cache[stem] = img
        if img is None:
            counts["image_unreadable"] += 1
            classified.append({**r, "match_type": "image_unreadable", "e2e_crop_path": ""})
            continue

        if match_type == "one_to_many_split":
            crop = crop_union_bbox(img, [detRects[d] for d in det_list])
        else:
            crop = crop_bbox(img, detRects[det_list[0]])
        if crop is None:
            counts["degenerate_det_box"] += 1
            classified.append({**r, "match_type": "degenerate_det_box", "e2e_crop_path": ""})
            continue

        lang_dir = os.path.join(CROP_OUT_ROOT, lang)
        os.makedirs(lang_dir, exist_ok=True)
        crop_path = os.path.join(lang_dir, r["instance_id"] + "_e2e.png")
        cv2.imwrite(crop_path, crop)
        counts[match_type] += 1
        classified.append({**r, "match_type": match_type, "e2e_crop_path": crop_path})

    print("Detection-match breakdown over the scoring universe:")
    for k, v in counts.items():
        print(f"  {k:22s} {v:5d} ({100*v/len(scored):.1f}%)")

    to_recognize = [c for c in classified
                     if c["match_type"] in ("one_to_one", "many_to_one_merged", "one_to_many_split")]
    print(f"\n{len(to_recognize)} instances have a usable detected-box crop "
          f"(one_to_one + many_to_one_merged + one_to_many_split) -- "
          f"running the {args.system} recognizer on them now.\n")

    e2e_text = {}

    if args.system == "clip":
        sys.path.insert(0, CLIP_REPO)
        import torch
        from inference import load_model, predict_batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        for lang in LANGUAGES:
            lang_items = [c for c in to_recognize if c["language"] == lang]
            if not lang_items:
                continue
            ckpt_path = os.path.join(CHECKPOINTS_DIR, lang, "best.pt")
            if not os.path.isfile(ckpt_path):
                print(f"[skip] {lang}: no checkpoint at {ckpt_path}")
                continue
            print(f"{lang}: recognizing {len(lang_items)} crops (checkpoint: {ckpt_path})")
            model, vocab = load_model(ckpt_path, CLIP_CKPT, args.max_len, device)
            for i in range(0, len(lang_items), args.batch_size):
                chunk = lang_items[i:i + args.batch_size]
                paths = [c["e2e_crop_path"] for c in chunk]
                try:
                    texts = predict_batch(model, vocab, paths, device, args.max_len,
                                           "beam", args.beam_size, use_amp=True,
                                           length_penalty=args.length_penalty)
                    for c, text in zip(chunk, texts):
                        text = (text or "").strip()
                        e2e_text[c["instance_id"]] = (text, "ok" if text else "no_text_detected")
                except Exception as e:
                    for c in chunk:
                        e2e_text[c["instance_id"]] = ("", f"exception:{e}")
                print(f"  [{min(i + args.batch_size, len(lang_items))}/{len(lang_items)}]")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    else:  # args.system == "vlm"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type != "cuda":
            print("!! WARNING: no CUDA device found -- 4-bit inference requires a GPU. This will likely fail.")

        adapter_dirs = {}
        for lang in LANGUAGES:
            adir = os.path.join(CHECKPOINTS_DIR, f"{lang}_qwen3vl_lora_visiontuned")
            if os.path.isdir(adir):
                adapter_dirs[lang] = adir
        if not adapter_dirs:
            raise SystemExit(f"No VLM adapters found under {CHECKPOINTS_DIR}/<lang>_qwen3vl_lora_visiontuned/")

        first_adapter_dir = next(iter(adapter_dirs.values()))
        print(f"Loading processor from {first_adapter_dir} ...")
        processor = AutoProcessor.from_pretrained(first_adapter_dir, trust_remote_code=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading base model {VLM_BASE_MODEL} in 4-bit ...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            VLM_BASE_MODEL, quantization_config=bnb_config,
            device_map={"": 0} if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        model = None

        @torch.no_grad()
        def vlm_generate(image):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            output_ids = model.generate(**inputs, max_new_tokens=args.vlm_max_new_tokens, do_sample=False)
            new_tokens = output_ids[0][input_len:]
            return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for lang in LANGUAGES:
            lang_items = [c for c in to_recognize if c["language"] == lang]
            if not lang_items:
                continue
            if lang not in adapter_dirs:
                print(f"[skip] {lang}: no VLM adapter at {CHECKPOINTS_DIR}/{lang}_qwen3vl_lora_visiontuned")
                continue
            if model is None:
                print(f"  attaching first LoRA adapter ({lang}) from {adapter_dirs[lang]} ...")
                model = PeftModel.from_pretrained(base_model, adapter_dirs[lang], adapter_name=lang)
            elif lang not in model.peft_config:
                print(f"  loading additional LoRA adapter ({lang}) from {adapter_dirs[lang]} ...")
                model.load_adapter(adapter_dirs[lang], adapter_name=lang)
            model.set_adapter(lang)
            model.eval()

            print(f"{lang}: recognizing {len(lang_items)} crops with the fine-tuned VLM")
            for i, c in enumerate(lang_items):
                try:
                    image = Image.open(c["e2e_crop_path"]).convert("RGB")
                    text = vlm_generate(image)
                    e2e_text[c["instance_id"]] = (text, "ok" if text else "no_text_detected")
                except Exception as e:
                    e2e_text[c["instance_id"]] = ("", f"exception:{e}")
                if (i + 1) % 20 == 0 or (i + 1) == len(lang_items):
                    print(f"  [{i + 1}/{len(lang_items)}]")

    n_total = len(scored)
    n_oracle_correct = 0
    n_e2e_correct = 0
    out_rows = []
    for c in classified:
        gcv = c.get("gcv_text", "")
        oracle_pred = c.get(oracle_col, "")
        oracle_correct = (oracle_pred == gcv)
        n_oracle_correct += oracle_correct

        e2e_pred, _ = e2e_text.get(c["instance_id"], ("", c["match_type"]))
        attempted = c["match_type"] in ("one_to_one", "many_to_one_merged", "one_to_many_split")
        e2e_correct = (attempted and e2e_pred == gcv)
        n_e2e_correct += e2e_correct

        out_rows.append({
            "instance_id": c["instance_id"], "language": c["language"], "gcv_text": gcv,
            "match_type": c["match_type"], "oracle_pred": oracle_pred,
            "oracle_correct": oracle_correct, "e2e_pred": e2e_pred, "e2e_correct": e2e_correct,
        })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{'='*70}")
    print(f" END-TO-END v2 (DETECT THEN RECOGNIZE) vs. ORACLE-CROP, system={args.system}")
    print(f"{'='*70}")
    print(f"Detector predictions used: {pred_path}")
    print(f"Scoring universe: {n_total} instances")
    print(f"Oracle-crop WRR: {100*n_oracle_correct/n_total:.2f}%  ({n_oracle_correct}/{n_total})")
    print(f"End-to-end WRR : {100*n_e2e_correct/n_total:.2f}%  ({n_e2e_correct}/{n_total})")
    print(f"Gap: {100*(n_oracle_correct-n_e2e_correct)/n_total:+.2f} pp")

    print(f"\nAccuracy by match_type (oracle vs. e2e, where e2e was attempted):")
    for t in ["one_to_one", "many_to_one_merged", "one_to_many_split", "missed"]:
        rs = [o for o in out_rows if o["match_type"] == t]
        if not rs:
            continue
        n = len(rs)
        oc = sum(o["oracle_correct"] for o in rs)
        ec = sum(o["e2e_correct"] for o in rs)
        print(f"  {t:22s} n={n:5d}  oracle_acc={100*oc/n:6.2f}%  e2e_acc={100*ec/n:6.2f}%")

    print(f"\nPer-language (end-to-end WRR vs oracle-crop WRR):")
    for lang in LANGUAGES:
        lang_rows = [o for o in out_rows if o["language"] == lang]
        if not lang_rows:
            continue
        n = len(lang_rows)
        oc = sum(o["oracle_correct"] for o in lang_rows)
        ec = sum(o["e2e_correct"] for o in lang_rows)
        print(f"  {lang:10s} n={n:4d}  oracle={100*oc/n:5.2f}%  e2e={100*ec/n:5.2f}%  gap={100*(oc-ec)/n:+5.2f}pp")

    print(f"\nSaved detailed per-instance results to: {args.out_csv}")
    print("Paste the full console output back to fold this into the paper.")


if __name__ == "__main__":
    main()
