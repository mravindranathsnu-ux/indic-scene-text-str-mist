#!/usr/bin/env python3
"""
run_track2_vlm.py

Track 2: runs the per-language Qwen3-VL-2B LoRA checkpoints fine-tuned on
Phase 1 / IndicSTR12 (phase1_vlm_lora_train.py, one adapter per language
under ~/multilingual-str-clip/checkpoints/<lang>_qwen3vl_lora_visiontuned/)
on the SAME MIST recognition_gt crops used for every other baseline
(Tesseract, PaddleOCR, EasyOCR, GPT-4V, Claude, Qwen3-VL zero-shot,
Track1-CLIP) -- adding a track2_text/track2_status column pair to the SAME
manifest.csv, ready for the same scoring pass against the Google Vision
pseudo-GT (gcv_text).

WHY THIS EXISTS
This is the direct follow-on to Track 1 (CLIP recognizer transferred to
MIST with zero MIST-specific training) and to the Qwen3-VL zero-shot
baseline (6.80% WRR on MIST, far behind Track1-CLIP's 39.34%). Since then,
phase1_vlm_lora_train.py fine-tuned Qwen3-VL-2B per language directly on
Phase 1's own IndicSTR12 training data (single epoch, --tune_vision LoRA)
and, evaluated on Phase 1's OWN test split, came within 1.4 points of the
CLIP recognizer's mean WRR (81.42% vs. 82.85%) -- a much smaller gap than
the zero-shot number suggested. This script answers the remaining
question: does that IndicSTR12-level near-parity transfer to MIST, the
same way Track1-CLIP's IndicSTR12 checkpoints transferred to MIST largely
intact (82.85% Phase-1 mean WRR -> 39.34% MIST WRR is a big absolute drop,
but MIST is a much harder dataset for every system, so the interesting
comparison is Track 2 vs. Track 1 on MIST, not Track 2's Phase 1 number
vs. its own MIST number).

PROMPT CHOICE: unlike the zero-shot Qwen3-VL baseline (run_qwen3vl_baseline.py),
which uses the GPT-4V/Claude-style list-extraction prompt for apples-to-apples
comparison among ZERO-SHOT general-purpose systems, this script uses the
SAME single-word OCR prompt the LoRA adapters were actually trained on
("Read the text written in this image. Respond with only the exact text,
nothing else.") -- matching phase1_vlm_lora_eval.py's inference exactly,
since MIST's recognition_gt crops are single-word crops (same convention
as IndicSTR12/Track1-CLIP), and evaluating a fine-tuned single-word OCR
model with a multi-word list-extraction prompt it never saw during
training would be measuring the wrong thing.

CHECKPOINT SELECTION
Uses checkpoints/<lang>_qwen3vl_lora_visiontuned/ under multilingual-str-clip
for exactly the 10 languages phase1_vlm_lora_train.py covers -- the same 10
languages Track 1 uses (assamese, meitei_manipuri, urdu have no Phase 1
checkpoint of either kind and are skipped, same as Track 1).

MEMORY/LOADING STRATEGY
The base model (Qwen3-VL-2B-Instruct, 4-bit NF4) is loaded ONCE and kept
resident; each language's LoRA adapter is loaded onto it via
PeftModel.load_adapter(..., adapter_name=lang) and activated with
set_adapter(lang) before that language's crops run, then left loaded (LoRA
adapters at rank 16 are a few tens of MB each in fp16/bf16 -- all 10
loaded simultaneously is negligible next to the 7.6GB budget the base
4-bit model itself needs). This avoids reloading the ~2B-parameter base
model 10 times, unlike run_track1_clip.py's per-language CLIP reload
(CLIP is small enough that reloading it 10 times is cheap; the VLM is not).

Usage:
    conda activate base
    cd ~/multilingual-str-clip   (adapters are found relative to here)
    export HF_HUB_OFFLINE=1      (adapters + base model already cached locally)
    python3 run_track2_vlm.py --manifest ~/mist_det_work/recognition_gt/manifest.csv
    # or one language while testing:
    python3 run_track2_vlm.py --manifest ~/mist_det_work/recognition_gt/manifest.csv --language telugu
"""
import argparse
import csv
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_MANIFEST = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")
DEFAULT_CHECKPOINTS_DIR = "checkpoints"  # relative to cwd (~/multilingual-str-clip)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

LANGUAGES = [
    "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "odia", "punjabi", "tamil", "telugu",
]

PROMPT = "Read the text written in this image. Respond with only the exact text, nothing else."


def load_rows(manifest_path):
    with open(manifest_path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--checkpoints_dir", default=DEFAULT_CHECKPOINTS_DIR,
                     help="Directory containing <lang>_qwen3vl_lora_visiontuned/ adapter folders. "
                          "Default assumes you're running from ~/multilingual-str-clip.")
    ap.add_argument("--base_model_id", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--language", choices=LANGUAGES, default=None,
                     help="Run just one language instead of all 10.")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--checkpoint_every", type=int, default=20)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("!! WARNING: no CUDA device found -- 4-bit inference requires a GPU. This will likely fail.")

    if not os.path.exists(args.manifest):
        raise SystemExit(f"No manifest at {args.manifest}")

    languages = [args.language] if args.language else LANGUAGES

    # Confirm every requested language has an adapter BEFORE loading the
    # (slow-ish) base model, so a typo/missing checkpoint fails fast.
    adapter_dirs = {}
    for lang in languages:
        adir = os.path.join(args.checkpoints_dir, f"{lang}_qwen3vl_lora_visiontuned")
        if os.path.isdir(adir):
            adapter_dirs[lang] = adir
        else:
            print(f"[skip] {lang}: no adapter at {adir}")
    if not adapter_dirs:
        raise SystemExit("No adapters found for any requested language -- check --checkpoints_dir.")

    rows = load_rows(args.manifest)
    fieldnames = list(rows[0].keys())
    text_col, status_col = "track2_text", "track2_status"
    if text_col not in fieldnames:
        fieldnames += [text_col, status_col]
    for row in rows:
        row.setdefault(text_col, "")
        row.setdefault(status_col, "pending")

    def save():
        tmp_path = args.manifest + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, args.manifest)

    # Load the processor from the FIRST adapter dir (phase1_vlm_lora_train.py
    # saves the processor alongside the adapter, matching phase1_vlm_lora_eval.py's
    # convention) -- the tokenizer/image-processor config is identical across all
    # per-language adapters since they all fine-tune the same base model.
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
    print(f"Loading base model {args.base_model_id} in 4-bit ...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    model = None  # becomes a PeftModel once the first adapter is loaded

    @torch.no_grad()
    def generate_prediction(image):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = output_ids[0][input_len:]
        text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    for lang, adapter_dir in adapter_dirs.items():
        lang_rows = [r for r in rows if r["language"] == lang]
        todo = [r for r in lang_rows if r[status_col] not in ("ok", "no_text_detected")]
        print(f"{lang}: {len(lang_rows) - len(todo)} already done, {len(todo)} remaining "
              f"(of {len(lang_rows)} total).")
        if not todo:
            continue

        if model is None:
            print(f"  attaching first LoRA adapter ({lang}) from {adapter_dir} ...")
            model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name=lang)
        elif lang not in model.peft_config:
            print(f"  loading additional LoRA adapter ({lang}) from {adapter_dir} ...")
            model.load_adapter(adapter_dir, adapter_name=lang)
        model.set_adapter(lang)
        model.eval()

        for i, row in enumerate(todo):
            try:
                image = Image.open(row["crop_path"]).convert("RGB")
                text = generate_prediction(image)
                row[text_col] = text
                row[status_col] = "ok" if text else "no_text_detected"
            except Exception as e:
                row[text_col], row[status_col] = "", f"exception:{e}"
            if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(todo):
                print(f"  [{i + 1}/{len(todo)}]")
                save()

    save()
    statuses = {}
    for r in rows:
        statuses[r[status_col]] = statuses.get(r[status_col], 0) + 1
    print(f"\nDone. Status breakdown: {statuses}")
    print(f"Manifest updated: {args.manifest}")


if __name__ == "__main__":
    main()
