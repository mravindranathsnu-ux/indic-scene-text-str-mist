#!/usr/bin/env python3
"""
deteval_score.py
Score saved MIST detection predictions (from run_full_test_inference.py)
using the canonical DetEval algorithm -- Wolf & Jolion one-to-one /
one-to-many / many-to-one REGION matching with proper don't-care
handling -- the actual protocol scene-text detection papers report
under, NOT MMOCR's simplified HmeanIOUMetric (plain greedy IoU>0.5
matching, plus an internal score-threshold sweep that reports the best
of several operating points rather than one fixed one).

The DetectionDetEvalEvaluator class below is copied VERBATIM from the
reference implementation used across the scene-text-detection
literature: concern/icdar2015_eval/detection/deteval.py, as shipped in
Megvii-CSG/MegReader and (as iou.py's plain-IoU sibling) MhLiao/DB --
the same lineage DBNet++ itself comes from. This is deliberate: getting
the one-to-many/many-to-one matching logic right from scratch is easy to
get subtly wrong, so this reuses the field's actual reference code
rather than reinventing it.

--tr/--tp below default to 0.7/0.6, from earlier (unconfirmed-from-
primary-source) research on the MIST paper's stated protocol. Also try
--tr 0.8 --tp 0.4, the ORIGINAL ICDAR DetEval defaults, as a sanity
cross-check -- if the paper didn't state custom values, this is what
"DetEval" alone usually means.

Usage:
    python3 deteval_score.py --tr 0.7 --tp 0.6
    python3 deteval_score.py --tr 0.8 --tp 0.4

Needs shapely (likely already installed as an mmocr dependency):
    pip install shapely --break-system-packages
"""
import argparse
import json
import os

import numpy as np
from shapely.geometry import Polygon

GT_DIR = os.path.expanduser("~/mist/gt/Test")
PRED_PATH = os.path.expanduser("~/mist_det_work/full_test_predictions.json")


class DetectionDetEvalEvaluator(object):
    """Verbatim from concern/icdar2015_eval/detection/deteval.py (MhLiao/DB
    lineage) -- see module docstring."""

    def __init__(
        self,
        area_recall_constraint=0.8, area_precision_constraint=0.4,
        ev_param_ind_center_diff_thr=1,
        mtype_oo_o=1.0, mtype_om_o=0.8, mtype_om_m=1.0
    ):
        self.area_recall_constraint = area_recall_constraint
        self.area_precision_constraint = area_precision_constraint
        self.ev_param_ind_center_diff_thr = ev_param_ind_center_diff_thr
        self.mtype_oo_o = mtype_oo_o
        self.mtype_om_o = mtype_om_o
        self.mtype_om_m = mtype_om_m

    def evaluate_image(self, gt, pred):
        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def one_to_one_match(row, col):
            cont = 0
            for j in range(len(recallMat[0])):
                if recallMat[row, j] >= self.area_recall_constraint and precisionMat[row, j] >= self.area_precision_constraint:
                    cont += 1
            if cont != 1:
                return False
            cont = 0
            for i in range(len(recallMat)):
                if recallMat[i, col] >= self.area_recall_constraint and precisionMat[i, col] >= self.area_precision_constraint:
                    cont += 1
            if cont != 1:
                return False
            if recallMat[row, col] >= self.area_recall_constraint and precisionMat[row, col] >= self.area_precision_constraint:
                return True
            return False

        def num_overlaps_gt(gtNum):
            cont = 0
            for detNum in range(len(detRects)):
                if detNum not in detDontCareRectsNum:
                    if recallMat[gtNum, detNum] > 0:
                        cont += 1
            return cont

        def num_overlaps_det(detNum):
            cont = 0
            for gtNum in range(len(recallMat)):
                if gtNum not in gtDontCareRectsNum:
                    if recallMat[gtNum, detNum] > 0:
                        cont += 1
            return cont

        def is_single_overlap(row, col):
            return num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1

        def one_to_many_match(gtNum):
            many_sum = 0
            detRectsL = []
            for detNum in range(len(recallMat[0])):
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                    if precisionMat[gtNum, detNum] >= self.area_precision_constraint:
                        many_sum += recallMat[gtNum, detNum]
                        detRectsL.append(detNum)
            if round(many_sum, 4) >= self.area_recall_constraint:
                return True, detRectsL
            return False, []

        def many_to_one_match(detNum):
            many_sum = 0
            gtRectsL = []
            for gtNum in range(len(recallMat)):
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                    if recallMat[gtNum, detNum] >= self.area_recall_constraint:
                        many_sum += precisionMat[gtNum, detNum]
                        gtRectsL.append(gtNum)
            if round(many_sum, 4) >= self.area_precision_constraint:
                return True, gtRectsL
            return False, []

        def center_distance(r1, r2):
            return ((np.mean(r1, axis=0) - np.mean(r2, axis=0)) ** 2).sum() ** 0.5

        def diag(r):
            r = np.array(r)
            return ((r[:, 0].max() - r[:, 0].min()) ** 2 + (r[:, 1].max() - r[:, 1].min()) ** 2) ** 0.5

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtDontCareRectsNum = []
        detDontCareRectsNum = []

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        for n in range(len(gt)):
            points = gt[n]['points']
            dontCare = gt[n]['ignore']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue
            gtRects.append(points)
            if dontCare:
                gtDontCareRectsNum.append(len(gtRects) - 1)

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue
            detRects.append(points)
            if len(gtDontCareRectsNum) > 0:
                for dontCareRectNum in gtDontCareRectsNum:
                    dontCareRect = gtRects[dontCareRectNum]
                    intersected_area = get_intersection(dontCareRect, points)
                    rdDimensions = Polygon(points).area
                    precision_dc = 0 if rdDimensions == 0 else intersected_area / rdDimensions
                    if precision_dc > self.area_precision_constraint:
                        detDontCareRectsNum.append(len(detRects) - 1)
                        break

        if len(gtRects) == 0:
            recall = 1
            precision = 0 if len(detRects) > 0 else 1

        if len(detRects) > 0 and len(gtRects) > 0:
            outputShape = [len(gtRects), len(detRects)]
            recallMat = np.empty(outputShape)
            precisionMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtRects), np.int8)
            detRectMat = np.zeros(len(detRects), np.int8)
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    rG = gtRects[gtNum]
                    rD = detRects[detNum]
                    intersected_area = get_intersection(rG, rD)
                    rgDimensions = Polygon(rG).area
                    rdDimensions = Polygon(rD).area
                    recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else intersected_area / rgDimensions
                    precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum:
                        if one_to_one_match(gtNum, detNum):
                            if is_single_overlap(gtNum, detNum):
                                rG = gtRects[gtNum]
                                rD = detRects[detNum]
                                normDist = center_distance(rG, rD)
                                normDist /= diag(rG) + diag(rD)
                                normDist *= 2.0
                                if normDist < self.ev_param_ind_center_diff_thr:
                                    gtRectMat[gtNum] = 1
                                    detRectMat[detNum] = 1
                                    recallAccum += self.mtype_oo_o
                                    precisionAccum += self.mtype_oo_o

            for gtNum in range(len(gtRects)):
                if gtNum not in gtDontCareRectsNum:
                    match, matchesDet = one_to_many_match(gtNum)
                    if match:
                        if num_overlaps_gt(gtNum) >= 2:
                            gtRectMat[gtNum] = 1
                            recallAccum += (self.mtype_oo_o if len(matchesDet) == 1 else self.mtype_om_o)
                            precisionAccum += (self.mtype_oo_o if len(matchesDet) == 1 else self.mtype_om_o * len(matchesDet))
                            for detNum in matchesDet:
                                detRectMat[detNum] = 1

            for detNum in range(len(detRects)):
                if detNum not in detDontCareRectsNum:
                    match, matchesGt = many_to_one_match(detNum)
                    if match:
                        if num_overlaps_det(detNum) >= 2:
                            detRectMat[detNum] = 1
                            recallAccum += (self.mtype_oo_o if len(matchesGt) == 1 else self.mtype_om_m * len(matchesGt))
                            precisionAccum += (self.mtype_oo_o if len(matchesGt) == 1 else self.mtype_om_m)
                            for gtNum in matchesGt:
                                gtRectMat[gtNum] = 1

            numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if len(detRects) > 0 else float(1)
            else:
                recall = float(recallAccum) / numGtCare
                precision = float(0) if (len(detRects) - len(detDontCareRectsNum)) == 0 else float(precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
            hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        numGtCare = len(gtRects) - len(gtDontCareRectsNum)
        numDetCare = len(detRects) - len(detDontCareRectsNum)

        return {
            'precision': precision, 'recall': recall, 'hmean': hmean,
            'gtCare': numGtCare, 'detCare': numDetCare,
            'recallAccum': recallAccum, 'precisionAccum': precisionAccum,
        }

    def combine_results(self, results):
        numGt = 0
        numDet = 0
        methodRecallSum = 0
        methodPrecisionSum = 0
        for result in results:
            numGt += result['gtCare']
            numDet += result['detCare']
            methodRecallSum += result['recallAccum']
            methodPrecisionSum += result['precisionAccum']
        methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
        methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        return {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}


def load_gt(stem):
    fp = os.path.join(GT_DIR, stem + ".json")
    if not os.path.exists(fp):
        return None
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    gt = []
    for shape in data["shapes"]:
        gt.append({"points": shape["points"], "ignore": shape.get("care", 0) == 0})
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr", type=float, default=0.7, help="area recall constraint")
    ap.add_argument("--tp", type=float, default=0.6, help="area precision constraint")
    ap.add_argument("--pred", type=str, default=PRED_PATH,
                     help="path to predictions json (default: original epoch-40 run)")
    args = ap.parse_args()
    pred_path = os.path.expanduser(args.pred)

    if not os.path.exists(pred_path):
        raise SystemExit(f"!! Missing {pred_path} -- run run_full_test_inference*.py first.")

    with open(pred_path) as f:
        predictions = json.load(f)

    evaluator = DetectionDetEvalEvaluator(area_recall_constraint=args.tr, area_precision_constraint=args.tp)

    results = []
    skipped = 0
    for stem, preds in predictions.items():
        gt = load_gt(stem)
        if gt is None:
            skipped += 1
            continue
        pred = [{"points": p["points"]} for p in preds]
        results.append(evaluator.evaluate_image(gt, pred))

    metrics = evaluator.combine_results(results)
    print(f"\n{'=' * 60}")
    print(f" DetEval (tr={args.tr}, tp={args.tp}) over {len(results)} images ({skipped} skipped -- missing GT)")
    print(f" predictions: {pred_path}")
    print(f"{'=' * 60}")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"Hmean:     {metrics['hmean'] * 100:.2f}%")
    print(f"\nFor comparison -- paper's Table (DetEval): DP-DETR 62.70 (69.61P/57.04R), "
          f"TBPN 57.06, MixNet 56.27, DB++ 51.42 (72.84P/39.73R)")


if __name__ == "__main__":
    main()
