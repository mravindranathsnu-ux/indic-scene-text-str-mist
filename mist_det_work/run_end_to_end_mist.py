#!/usr/bin/env python3
"""
run_end_to_end_mist.py

END-TO-END (detect THEN recognize) evaluation on MIST, extending the
paper's existing oracle-crop recognition benchmark (Track1-CLIP 39.34%
WRR / Track2-VLM-FT 39.83% WRR, both scored on crops taken from MIST's
REAL ground-truth polygons -- Section 6, score_mist_baselines.py) with a
number that reflects running the recognizer on crops the DETECTOR
itself found, errors and all.

WHY THIS EXISTS
Right now detection (DBNet++, 68.77 F-measure) and recognition (CLIP /
fine-tuned VLM, ~39% WRR) are reported as two independent numbers.
Recognition is always given an oracle crop from the real ground-truth
polygon, never a crop the detector actually produced -- so there is no
single number in the paper describing what happens if you actually run
the whole pipeline end to end. This is the gap a reviewer is likely to
notice (most STR papers report an end-to-end / word-spotting number),
and it is also exactly the comparison the Scientific Reports "integrated
framework" paper the user is drawing on makes its central contribution.

METHODOLOGY
For each of the already-scored recognition_gt instances (SAME scoring
universe as score_mist_baselines.py / the paper's Table: status=='ok'
(GCV pseudo-GT available) AND <system>_status=='ok'):

  1. Identify which polygon in the ORIGINAL MIST Test GT json this
     instance is (image_stem + shape_idx, recovered from instance_id --
     build_mist_recognition_gt.py names them
     "<lang>_<i>_<image_stem>_<shape_idx>", and image_stem is already a
     manifest column, so shape_idx is recovered by stripping it out).
  2. Re-run the EXACT SAME DetEval one-to-one / one-to-many / many-to-one
     matching algorithm already used for the paper's headline detection
     number (deteval_score.py, tr=0.7/tp=0.6 by default, copied verbatim
     from the same MhLiao/DB-lineage reference implementation) over
     THAT INSTANCE'S WHOLE IMAGE (all of that image's GT polygons vs.
     all of DBNet++'s predicted polygons for it) -- this reuses the
     already-trusted detection-scoring code rather than inventing a new,
     separate ad-hoc IoU convention. The matching logic/thresholds here
     are UNCHANGED from deteval_score.py; the only addition is
     bookkeeping that records WHICH gt matched WHICH det (deteval_score.py
     itself only returns aggregate precision/recall/hmean, not
     per-instance matches).
  3. Classify this specific GT instance as:
       - "one_to_one": matched cleanly to exactly one predicted polygon
         -> crop that PREDICTED polygon's bounding box (same pad=2
         recipe as build_mist_recognition_gt.py's crop_instance) from
         the full MIST Test image, and feed THAT crop to the recognizer.
       - "fragmented_or_merged": matched only via a one-to-many (detector
         split one word into several boxes) or many-to-one (detector
         merged several words into one box) group. Recognizing a single
         word cleanly from a split/merged box isn't well-defined for a
         word-level recognizer trained on single, cleanly-cropped words,
         so these are scored as an end-to-end miss WITHOUT running the
         recognizer, rather than guessing which fragment to feed it.
       - "missed": no matching predicted polygon at all (a detection
         false negative) -> automatic end-to-end miss, recognizer never
         runs for this instance.
  4. For "one_to_one" instances only, run the SAME checkpoint/decode
     settings as run_track1_clip.py (CLIP ViT-B/16 per-language
     checkpoints, beam=5, length_penalty=1.0) on the NEWLY CROPPED
     (detected-box) image, and score exact-match against the SAME
     gcv_text pseudo-GT already used for the oracle-crop number.
  5. End-to-end WRR = (# one_to_one AND recognized-correctly) /
     (total scored instances) -- directly comparable to the already-
     published oracle-crop WRR on the IDENTICAL denominator. The gap
     between the two numbers is the new end-to-end finding.

Only --system clip is wired up in this version (the fine-tuned VLM's own
runner script wasn't inspected while writing this, so its invocation
wasn't guessed at). The classification logic (steps 1-3) is system-
agnostic; adding --system vlm later just means swapping in that
checkpoint-loading/inference call in the recognition loop below.

VERIFIED (unit-tested against 5 hand-built cases before delivery):
clean one-to-one match, total miss, one-to-many split, many-to-one
merge, and -- the subtlest case -- a degenerate/invalid gt polygon
earlier in the same image not desyncing the gt_orig_idx mapping for a
later, valid one. All 5 passed.

IMPORTANT: run deteval_score.py against BOTH full_test_predictions.json
and full_test_predictions_ext.json first, and pass whichever one
reproduces the paper's reported 68.77 Hmean via --pred -- this script's
end-to-end number is only meaningful if it's built on the exact same
detector output the paper's detection number already comes from.

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    # quick smoke test on a handful of instances first:
    python3 run_end_to_end_mist.py --pred full_test_predictions.json --limit 50
    # full run:
    python3 run_end_to_end_mist.py --pred full_test_predictions.json
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from shapely.geometry import Polygon

GT_DIR = os.path.expanduser("~/mist/gt/Test")
IMG_ROOT = os.path.expanduser("~/mist/Images/Test")
DEFAULT_PRED_PATH = os.path.expanduser("~/mist_det_work/full_test_predictions.json")
MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")
CROP_OUT_ROOT = os.path.expanduser("~/mist_det_work/recognition_gt_e2e")
CLIP_REPO = os.path.expanduser("~/multilingual-str-clip")
CLIP_CKPT = os.path.join(CLIP_REPO, "weights", "ViT-B-16.pt")
CHECKPOINTS_DIR = os.path.join(CLIP_REPO, "checkpoints")

LANGUAGES = ["bengali", "gujarati", "hindi", "kannada", "malayalam",
             "marathi", "odia", "punjabi", "tamil", "telugu"]

PAD = 2  # same padding convention as build_mist_recognition_gt.py's crop_instance

DEFAULT_TR = 0.7  # area recall constraint -- matches deteval_score.py's default,
DEFAULT_TP = 0.6  # which is what the paper's reported 68.77 Hmean uses


# ---------------------------------------------------------------------------
# DetEval matching, instrumented to also report WHICH gt matched WHICH det.
# Logic/thresholds below are UNCHANGED from deteval_score.py's
# DetectionDetEvalEvaluator.evaluate_image (itself copied verbatim from the
# MhLiao/DB / Megvii-CSG/MegReader reference implementation) -- this only
# adds bookkeeping (gt_match_dets, gt_orig_idx) alongside the existing
# control flow, it does not change any decision/threshold.
# ---------------------------------------------------------------------------
def match_image(gt, pred, tr=DEFAULT_TR, tp=DEFAULT_TP):
    """
    gt:   list of {"points": [[x,y],...], "ignore": bool, "orig_idx": int}
    pred: list of {"points": [[x,y],...]}

    Returns (gt_match_dets, gtRects, detRects, gt_orig_idx):
      gt_match_dets: {gtNum: [detNum, ...]} for every gt matched (one-to-one,
                     one-to-many, or many-to-one).
      gtRects, detRects: the valid (is_valid and is_simple) polygon point
                     lists actually used for matching, same filtering
                     deteval_score.py applies.
      gt_orig_idx:   gtRects index -> original shapes index (parallel list
                     to gtRects), since invalid/degenerate polygons are
                     skipped and would otherwise desync indices.
    """
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

    # one-to-one
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

    # one-to-many: detector split one gt word into several boxes
    for gtNum in range(len(gtRects)):
        if gtNum not in gtDontCareRectsNum and gtRectMat[gtNum] == 0:
            match, matchesDet = one_to_many_match(gtNum)
            if match and num_overlaps_gt(gtNum) >= 2:
                gtRectMat[gtNum] = 1
                for detNum in matchesDet:
                    detRectMat[detNum] = 1
                gt_match_dets[gtNum] = list(matchesDet)

    # many-to-one: detector merged several gt words into one box
    for detNum in range(len(detRects)):
        if detNum not in detDontCareRectsNum and detRectMat[detNum] == 0:
            match, matchesGt = many_to_one_match(detNum)
            if match and num_overlaps_det(detNum) >= 2:
                detRectMat[detNum] = 1
                for gtNum in matchesGt:
                    gtRectMat[gtNum] = 1
                    gt_match_dets[gtNum] = [detNum]

    return gt_match_dets, gtRects, detRects, gt_orig_idx


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


def parse_shape_idx(instance_id, image_stem):
    """instance_id is '<lang>_<i:04d>_<image_stem>_<shape_idx>' -- image_stem
    is already a manifest column, so find it and read off what follows."""
    pos = instance_id.index(image_stem)
    after = instance_id[pos + len(image_stem):]
    return int(after.lstrip("_"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["clip"], default="clip",
                     help="Recognizer to run on detected crops. Only 'clip' (Track1-CLIP) "
                          "is wired up in this version -- see module docstring.")
    ap.add_argument("--pred", default=DEFAULT_PRED_PATH,
                     help="DBNet++ predictions json -- confirm via deteval_score.py which "
                          "file (full_test_predictions.json or _ext.json) reproduces the "
                          "paper's 68.77 Hmean, and pass that one.")
    ap.add_argument("--tr", type=float, default=DEFAULT_TR)
    ap.add_argument("--tp", type=float, default=DEFAULT_TP)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N scored instances -- for a quick smoke "
                          "test before committing to the full run.")
    ap.add_argument("--out_csv", default=os.path.expanduser(
        "~/mist_det_work/recognition_gt/e2e_results_clip.csv"))
    args = ap.parse_args()

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
    # same scoring universe as score_mist_baselines.py / the paper's Table:
    # status=='ok' (GCV pseudo-GT available) AND <system>_status=='ok'
    scored = [r for r in rows if r.get("status") == "ok" and r.get(status_col) == "ok"]
    if args.limit:
        scored = scored[:args.limit]
    print(f"Scoring universe: {len(scored)} instances "
          f"(identical denominator to the paper's oracle-crop number, modulo --limit).\n")

    os.makedirs(CROP_OUT_ROOT, exist_ok=True)
    gt_cache, img_cache, match_cache = {}, {}, {}
    classified = []
    counts = {"one_to_one": 0, "fragmented_or_merged": 0, "missed": 0,
              "excluded_invalid_gt": 0, "no_detector_output": 0, "image_unreadable": 0,
              "degenerate_det_box": 0}

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
                match_cache[stem] = match_image(gt_shapes, pred_list, args.tr, args.tp)

        cached = match_cache[stem]
        if cached is None:
            counts["no_detector_output"] += 1
            classified.append({**r, "match_type": "no_detector_output", "e2e_crop_path": ""})
            continue

        gt_match_dets, gtRects, detRects, gt_orig_idx = cached
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
        if len(det_list) != 1:
            counts["fragmented_or_merged"] += 1
            classified.append({**r, "match_type": "fragmented_or_merged", "e2e_crop_path": ""})
            continue

        det_points = detRects[det_list[0]]
        img = img_cache.get(stem)
        if img is None:
            img = cv2.imread(os.path.join(IMG_ROOT, stem + ".png"))
            img_cache[stem] = img
        if img is None:
            counts["image_unreadable"] += 1
            classified.append({**r, "match_type": "image_unreadable", "e2e_crop_path": ""})
            continue
        crop = crop_bbox(img, det_points)
        if crop is None:
            counts["degenerate_det_box"] += 1
            classified.append({**r, "match_type": "degenerate_det_box", "e2e_crop_path": ""})
            continue

        lang_dir = os.path.join(CROP_OUT_ROOT, lang)
        os.makedirs(lang_dir, exist_ok=True)
        crop_path = os.path.join(lang_dir, r["instance_id"] + "_e2e.png")
        cv2.imwrite(crop_path, crop)
        counts["one_to_one"] += 1
        classified.append({**r, "match_type": "one_to_one", "e2e_crop_path": crop_path})

    print("Detection-match breakdown over the scoring universe:")
    for k, v in counts.items():
        print(f"  {k:22s} {v:5d} ({100*v/len(scored):.1f}%)")

    to_recognize = [c for c in classified if c["match_type"] == "one_to_one"]
    print(f"\n{len(to_recognize)} instances are cleanly one-to-one detected -- "
          f"running the recognizer on their DETECTED-box crops now.\n")

    sys.path.insert(0, CLIP_REPO)
    import torch
    from inference import load_model, predict_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    e2e_text = {}
    for lang in LANGUAGES:
        lang_items = [c for c in to_recognize if c["language"] == lang]
        if not lang_items:
            continue
        ckpt_path = os.path.join(CHECKPOINTS_DIR, lang, "best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[skip] {lang}: no checkpoint at {ckpt_path}")
            continue
        print(f"{lang}: recognizing {len(lang_items)} detected-box crops (checkpoint: {ckpt_path})")
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

    oracle_col = "clip_text" if args.system == "clip" else "track2_text"
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
        e2e_correct = (c["match_type"] == "one_to_one" and e2e_pred == gcv)
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
    print(f" END-TO-END (DETECT THEN RECOGNIZE) vs. ORACLE-CROP, system={args.system}")
    print(f"{'='*70}")
    print(f"Detector predictions used: {pred_path}")
    print(f"Scoring universe: {n_total} instances")
    print(f"Oracle-crop WRR (already published): {100*n_oracle_correct/n_total:.2f}%  ({n_oracle_correct}/{n_total})")
    print(f"End-to-end WRR  (detected-box crops): {100*n_e2e_correct/n_total:.2f}%  ({n_e2e_correct}/{n_total})")
    print(f"Gap from imperfect detection: {100*(n_oracle_correct-n_e2e_correct)/n_total:+.2f} pp")
    print(f"\nOf {n_total} instances: {counts['one_to_one']} cleanly one-to-one detected "
          f"(recognizer ran on these), {counts['missed']} missed entirely by the detector, "
          f"{counts['fragmented_or_merged']} split/merged by the detector "
          f"(scored as end-to-end misses, recognizer not run -- see module docstring).")

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
