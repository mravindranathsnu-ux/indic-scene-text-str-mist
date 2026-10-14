#!/usr/bin/env python3
"""
run_qwen3vl_baseline.py

Zero-shot Qwen3-VL baseline (NO training) on the same MIST recognition_gt
crops used for every other baseline -- adds a qwen3vl_text/qwen3vl_status
column pair to the SAME manifest.csv, ready for the same scoring pass
against the Google Vision pseudo-GT (gcv_text).

This is the "before we commit to real Track 2 training" step you chose:
an off-the-shelf Qwen3-VL model, evaluated the exact same way GPT-4V and
Claude were (same system+user prompt, same list-parsing convention), so
it slots straight into score_mist_baselines.py's SYSTEMS list alongside
everything else.

Why the 2B model by default
----------------------------
Your GPU driver is currently too old for the installed PyTorch build
(CUDA 12080 "too old"), so this -- like Track 1 -- will most likely run
on CPU unless you've since fixed the driver. Qwen3-VL-4B/8B on CPU would
be very slow for 3000 autoregressive generations; Qwen3-VL-2B-Instruct
is the right starting point for feasibility. Pass --model to try a
bigger one once you have GPU inference working, e.g.:
    --model Qwen/Qwen3-VL-4B-Instruct
    --model Qwen/Qwen3-VL-8B-Instruct

If CPU turns out to be too slow even for the 2B model (check the dry-run
timing below before committing to the full 3000), the next thing to try
is a quantized GGUF build through llama.cpp/Ollama rather than raw
transformers -- that's a different runtime and out of scope for this
script, flagging it here so you know the escape hatch exists.

Setup (do this in a FRESH env, not mist_det)
---------------------------------------------
Qwen3-VL needs a very new transformers (>=5.17.0 as of Sep 2026) --
newer than anything the rest of this project depends on, and nothing
else here uses transformers at all (CLIP/Track 1 uses its own loader,
GPT-4V/Claude use their own SDKs), so there's no reason to risk
destabilizing mist_det:
    conda create -n qwen3vl python=3.11 -y
    conda activate qwen3vl
    pip install --upgrade transformers accelerate torch pillow

Usage
-----
    conda activate qwen3vl
    cd ~/mist_det_work

    # ALWAYS do this first -- times a small sample and extrapolates
    # total runtime before you commit to all ~3000 crops:
    python3 run_qwen3vl_baseline.py --limit 20

    # then, once you've seen the estimate and you're happy with it:
    python3 run_qwen3vl_baseline.py

    # resumable/checkpointed like every other engine here -- Ctrl-C and
    # re-run picks up where it left off (via --limit, run in batches too)
"""
import argparse
import ast
import csv
import os
import re
import time

MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Same prompt convention as run_vlm_baselines.py (GPT-4V/Claude) -- keeps
# this an apples-to-apples comparison, not a differently-elicited result.
SYSTEM_PROMPT = "You are an OCR system that extracts text from images."
USER_PROMPT = (
    "Extract all text from this image. Return all words in a list "
    "format like this: ['word1', 'word2']"
)


def parse_list_response(text):
    """Same parser as run_vlm_baselines.py, including the fix for the
    apostrophe-in-refusal-prose bug found while scoring GPT-4V/Claude:
    the quoted-substring fallback only fires when the response actually
    looks list-shaped (starts with '['), never on ordinary prose."""
    text = (text or "").strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return " ".join(str(x) for x in parsed).strip()
    except (ValueError, SyntaxError):
        pass
    if text.startswith("["):
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        if quoted:
            words = [a or b for a, b in quoted]
            return " ".join(w for w in words if w).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                     help=f"Default: {DEFAULT_MODEL}. Bump to 4B/8B once "
                          "GPU inference is working.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                     help="Process at most this many remaining rows, then "
                          "stop and print a timing estimate for the full "
                          "run. Use this FIRST (e.g. --limit 20) before "
                          "committing to all ~3000 crops -- a multi-"
                          "billion-parameter VLM on CPU can plausibly take "
                          "many hours where Track 1's much smaller CLIP "
                          "model did not.")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  (CPU fallback -- this is expected until the NVIDIA driver "
              "is updated; the --limit dry run below tells you what that "
              "means for total runtime)")

    print(f"Loading {args.model} ...")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    text_col, status_col = "qwen3vl_text", "qwen3vl_status"
    if text_col not in fieldnames:
        fieldnames += [text_col, status_col]
    for row in rows:
        row.setdefault(text_col, "")
        row.setdefault(status_col, "pending")

    todo = [r for r in rows if r[status_col] not in ("ok", "no_text_detected")]
    print(f"qwen3vl ({args.model}): {len(rows) - len(todo)} already done, "
          f"{len(todo)} remaining.")
    if args.limit is not None:
        todo = todo[:args.limit]
        print(f"--limit set: processing only {len(todo)} rows this run.")

    def checkpoint():
        # Same atomic + encoding-safe pattern as every other baseline
        # script in this project -- write to a sibling .tmp file, then
        # os.replace() over the real manifest, so a crash mid-write can
        # never truncate/corrupt manifest.csv (this is exactly the bug
        # that cost us 1600+ rows during the GPT-4V run).
        tmp_path = MANIFEST_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, MANIFEST_PATH)

    per_image_times = []
    start = time.time()
    for i, row in enumerate(todo):
        t0 = time.time()
        try:
            img = Image.open(row["crop_path"]).convert("RGB")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens)
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            text = parse_list_response(raw)
            row[text_col] = text
            row[status_col] = "ok" if text else "no_text_detected"
        except Exception as e:
            row[text_col], row[status_col] = "", f"exception:{e}"

        per_image_times.append(time.time() - t0)
        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(todo):
            elapsed = time.time() - start
            avg = sum(per_image_times) / len(per_image_times)
            print(f"  [{i + 1}/{len(todo)}] ({elapsed:.0f}s elapsed, "
                  f"{avg:.2f}s/image avg)")
            checkpoint()

    checkpoint()
    statuses = {}
    for r in rows:
        statuses[r[status_col]] = statuses.get(r[status_col], 0) + 1
    print(f"\nqwen3vl done. Status breakdown: {statuses}")
    print(f"Manifest updated: {MANIFEST_PATH}")

    if args.limit is not None and per_image_times:
        avg = sum(per_image_times) / len(per_image_times)
        remaining_after_this = len(rows) - sum(statuses.values()) + statuses.get("pending", 0)
        # Rows still not "ok"/"no_text_detected"/exception after this run:
        still_todo = sum(
            1 for r in rows if r[status_col] not in ("ok", "no_text_detected")
            and not str(r[status_col]).startswith("exception")
        )
        est_full_hours = avg * 3000 / 3600
        est_remaining_hours = avg * still_todo / 3600
        print(f"\n--- Timing estimate (from this {len(per_image_times)}-image sample) ---")
        print(f"  {avg:.2f}s/image average")
        print(f"  Extrapolated full 3000-crop run: ~{est_full_hours:.1f} hours")
        print(f"  Remaining rows not yet done ({still_todo}): ~{est_remaining_hours:.1f} hours")
        print("  If that's too slow: try a smaller/quantized model, fix the "
              "GPU driver, or run in batches over multiple sessions -- this "
              "script is fully resumable either way.")


if __name__ == "__main__":
    main()
