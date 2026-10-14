#!/usr/bin/env python3
"""
phase1_vlm_lora_eval.py -- WRR/CRR evaluation for a phase1_vlm_lora_train.py
LoRA adapter, scored on Phase 1's own IndicSTR12 test split (the same split
Table 1 of the IJDAR paper reports PARSeq-published and CLIP-this-work
numbers against), so the new VLM number is directly comparable in the same
table row.

Generalized from bstd_vlm_lora_eval.py (same scoring logic, unchanged) --
only the per-language comparison numbers below are new, pulled straight
from the IJDAR paper's Table 1 (Phase 1: CLIP ViT-B/16 vs. published
PARSeq baseline on IndicSTR12).

WHY NORMALIZE BEFORE SCORING
Both prediction and ground truth are normalized to Unicode NFC before
comparison -- guards against legacy-decomposed-vowel-sign label encoding
quirks the same way bstd_vlm_lora_eval.py did for BSTD Kannada, and the
same class of issue this project's own error analysis found in Telugu/
Kannada GT digit encoding.

WHAT THIS DOES
  - Loads the base VLM in 4-bit + attaches the trained LoRA adapter.
  - Greedy-decodes a prediction for each test image (no beam search, no
    lexicon reranking -- a clean baseline matching phase1_vlm_lora_train.py's
    scope; add decode-time levers later only if this baseline earns it).
  - Reports WRR (exact-match rate) and CRR (1 - normalized Levenshtein
    character error rate, clipped at 0) over the test set.
  - Writes a predictions CSV (image_path, ground_truth, prediction,
    exact_match).

IMPORTANT CAVEATS
  - Needs: pip install --break-system-packages transformers accelerate
    peft bitsandbytes pillow -- same environment as training.
  - Batch size is fixed at 1 (one image at a time), same reasoning as
    bstd_vlm_lora_eval.py: correctness over speed for Qwen-VL's dynamic
    resolution + position-id interactions.

Usage (smoke test first):
  python3 phase1_vlm_lora_eval.py \\
      --base_model_id Qwen/Qwen3-VL-2B-Instruct \\
      --adapter_dir checkpoints/telugu_qwen3vl_lora_visiontuned \\
      --test_tsv data/telugu/test/test.tsv \\
      --image_root data/telugu/test/images \\
      --output_csv results/telugu_qwen3vl_lora_predictions.csv \\
      --language telugu \\
      --max_eval_samples 10
"""
import argparse
import csv
import json
import os
import unicodedata

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

PROMPT = "Read the text written in this image. Respond with only the exact text, nothing else."

# Table 1 of the IJDAR paper: (PARSeq published WRR, PARSeq published CRR,
# CLIP ViT-B/16 this-work WRR, CLIP ViT-B/16 this-work CRR), all percent.
PHASE1_TABLE1 = {
    "bengali":   (62.04, 83.08, 82.40, 93.26),
    "gujarati":  (60.61, 85.02, 74.59, 88.03),
    "hindi":     (45.14, 76.01, 83.66, 93.84),
    "kannada":   (63.57, 88.64, 89.66, 96.46),
    "malayalam": (68.81, 90.10, 92.54, 97.83),
    "marathi":   (63.50, 86.74, 88.12, 96.31),
    "odia":      (71.30, 89.13, 84.92, 94.77),
    "punjabi":   (78.70, 92.68, 79.62, 92.72),
    "tamil":     (67.35, 87.56, 75.24, 92.29),
    "telugu":    (71.94, 92.18, 77.74, 92.04),
}


def load_rows(test_tsv, image_root, max_samples=None):
    rows = []
    with open(test_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1]:
                rows.append((parts[0], parts[1]))
    samples = []
    for rel_path, label in rows:
        full_path = rel_path if os.path.isfile(rel_path) else os.path.join(image_root, os.path.basename(rel_path))
        if not os.path.isfile(full_path):
            continue
        samples.append((full_path, label))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


@torch.no_grad()
def generate_prediction(model, processor, image, device, max_new_tokens=32):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)

    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[0][input_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_report", default=None,
                     help="Optional JSON report path (mirrors the CLIP pipeline's *_test_report.json "
                          "schema loosely, for easy side-by-side reading).")
    ap.add_argument("--language", default=None, choices=sorted(PHASE1_TABLE1),
                     help="If given, prints the PARSeq/CLIP comparison row from Table 1 for this language.")
    ap.add_argument("--max_eval_samples", type=int, default=None, help="Smoke-test with e.g. 10 before a full run.")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! WARNING: no CUDA device found -- 4-bit inference requires a GPU. This will likely fail.")

    print(f"Loading processor + base model {args.base_model_id} ...")
    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    print(f"Attaching LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    samples = load_rows(args.test_tsv, args.image_root, max_samples=args.max_eval_samples)
    print(f"Evaluating on {len(samples)} samples"
          + (f" (smoke test, --max_eval_samples={args.max_eval_samples})" if args.max_eval_samples else "") + ".")
    if len(samples) < 1:
        raise SystemExit("No valid (image, label) pairs found -- check --test_tsv/--image_root.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    n_exact = 0
    total_cer_numer = 0.0
    total_cer_denom = 0
    rows_out = []

    for i, (image_path, gt_raw) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")
        pred_raw = generate_prediction(model, processor, image, device, max_new_tokens=args.max_new_tokens)

        gt = unicodedata.normalize("NFC", gt_raw.strip())
        pred = unicodedata.normalize("NFC", pred_raw.strip())

        exact_match = (pred == gt)
        n_exact += int(exact_match)
        dist = levenshtein(pred, gt)
        total_cer_numer += dist
        total_cer_denom += max(len(gt), 1)

        rows_out.append({
            "image_path": image_path,
            "ground_truth": gt,
            "prediction": pred,
            "exact_match": exact_match,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            running_wrr = n_exact / (i + 1)
            print(f"  [{i + 1}/{len(samples)}] running WRR={running_wrr:.4f}")

    wrr = n_exact / len(samples)
    cer = total_cer_numer / total_cer_denom if total_cer_denom else float("nan")
    crr = max(0.0, 1.0 - cer)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "prediction", "exact_match"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n{'='*70}\n RESULTS (n={len(samples)})\n{'='*70}")
    print(f"WRR (exact match): {wrr:.4f} ({n_exact}/{len(samples)})")
    print(f"CRR (1 - char error rate): {crr:.4f}")
    print(f"Predictions written to {args.output_csv}")

    if args.language and args.language in PHASE1_TABLE1:
        parseq_wrr, parseq_crr, clip_wrr, clip_crr = PHASE1_TABLE1[args.language]
        print(f"\nFor comparison ({args.language}, Table 1 of the IJDAR paper):")
        print(f"  PARSeq (published):        WRR={parseq_wrr:.2f}%  CRR={parseq_crr:.2f}%")
        print(f"  CLIP ViT-B/16 (this work):  WRR={clip_wrr:.2f}%  CRR={clip_crr:.2f}%")
        print(f"  Qwen3-VL-2B LoRA (this run): WRR={wrr*100:.2f}%  CRR={crr*100:.2f}%")

    if args.output_report:
        report = {
            "language": args.language,
            "adapter_dir": args.adapter_dir,
            "test_tsv": args.test_tsv,
            "num_samples": len(samples),
            "metrics": {"WRR": wrr, "CRR": crr, "CER": cer},
        }
        if args.language and args.language in PHASE1_TABLE1:
            parseq_wrr, parseq_crr, clip_wrr, clip_crr = PHASE1_TABLE1[args.language]
            report["comparison"] = {
                "parseq_published_wrr_pct": parseq_wrr,
                "parseq_published_crr_pct": parseq_crr,
                "clip_this_work_wrr_pct": clip_wrr,
                "clip_this_work_crr_pct": clip_crr,
            }
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report written to {args.output_report}")


if __name__ == "__main__":
    main()
