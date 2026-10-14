#!/usr/bin/env python3
"""
run_track1_clip.py

Track 1: runs the already-trained Phase 1 CLIP-based STR recognizers
(multilingual-str-clip, one checkpoint per language) on the SAME MIST
recognition_gt crops used for every other baseline (Tesseract,
PaddleOCR, EasyOCR, GPT-4V, Claude) -- adding a clip_text/clip_status
column pair to the SAME manifest.csv, ready for the same scoring pass
against the Google Vision pseudo-GT (gcv_text).

Reuses multilingual-str-clip's own inference.py (load_model +
predict_batch) rather than reimplementing model loading -- that's
also why this must run inside the mist_det conda env with
~/multilingual-str-clip on the path (added below) so `import torch`,
`vocab.py`, `model.py` etc. resolve the same way they do for that
project's own scripts.

Checkpoint selection (confirmed Sep 2026, from training_summary.json
+ test_report.json across every checkpoints/<lang>/ and
checkpoints/<lang>_backups/*/ directory):
  - Each language's top-level checkpoints/<language>/best.pt is used
    directly. For tamil, telugu, and punjabi the extra *_backups/*
    folders are byte-identical copies of this same checkpoint from
    different experiment milestones (confirmed via matching file
    size AND identical test_report.json metrics/timestamps) -- not
    real alternatives.
  - Gujarati's top-level best.pt is confirmed to be the LATER,
    better-performing checkpoint (test WRR .746, CRR .880) vs. the
    misleadingly-named "current_best" backup, which is actually an
    earlier, worse checkpoint (WRR .405) from before further tuning
    + lexicon reranking produced the current one.
  - assamese, meitei_manipuri, urdu have no trained checkpoint at
    all (just a placeholder README.txt) and are skipped -- MIST and
    Phase 1 share exactly the other 10 languages.

Decode settings (beam=5, length_penalty=1.0, max_len=32, batch=32)
match the parameters recorded in this project's own test_report.json
files, for consistency with the numbers already reported for Phase 1.

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    python3 run_track1_clip.py
    # or one language while testing:
    python3 run_track1_clip.py --language tamil
    # lower --batch_size if you hit a CUDA out-of-memory error
    # (7.6GB usable VRAM) -- unlikely at these crop counts/sizes,
    # but the checkpoints load one language at a time and are freed
    # (torch.cuda.empty_cache()) before the next language loads.
"""
import argparse
import csv
import os
import sys

MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")
CLIP_REPO = os.path.expanduser("~/multilingual-str-clip")
CLIP_CKPT = os.path.join(CLIP_REPO, "weights", "ViT-B-16.pt")
CHECKPOINTS_DIR = os.path.join(CLIP_REPO, "checkpoints")

LANGUAGES = [
    "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "odia", "punjabi", "tamil", "telugu",
]

DEFAULT_DECODE = "beam"
DEFAULT_BEAM_SIZE = 5
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_MAX_LEN = 32
DEFAULT_BATCH_SIZE = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=LANGUAGES, default=None,
                     help="Run just one language instead of all 10.")
    ap.add_argument("--decode", choices=["greedy", "beam"], default=DEFAULT_DECODE)
    ap.add_argument("--beam_size", type=int, default=DEFAULT_BEAM_SIZE)
    ap.add_argument("--length_penalty", type=float, default=DEFAULT_LENGTH_PENALTY)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = ap.parse_args()

    sys.path.insert(0, CLIP_REPO)
    import torch
    from inference import load_model, predict_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"No manifest at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    if "clip_text" not in fieldnames:
        fieldnames += ["clip_text", "clip_status"]
    for row in rows:
        row.setdefault("clip_text", "")
        row.setdefault("clip_status", "pending")

    def save():
        tmp_path = MANIFEST_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, MANIFEST_PATH)

    languages = [args.language] if args.language else LANGUAGES
    for lang in languages:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, lang, "best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[skip] {lang}: no checkpoint at {ckpt_path}")
            continue

        lang_rows = [r for r in rows if r["language"] == lang]
        todo = [r for r in lang_rows if r["clip_status"] not in ("ok", "no_text_detected")]
        print(f"{lang}: {len(lang_rows) - len(todo)} already done, {len(todo)} remaining "
              f"(of {len(lang_rows)} total).")
        if not todo:
            continue

        print(f"  loading checkpoint: {ckpt_path}")
        model, vocab = load_model(ckpt_path, CLIP_CKPT, args.max_len, device)

        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            paths = [r["crop_path"] for r in chunk]
            try:
                texts = predict_batch(model, vocab, paths, device, args.max_len,
                                       args.decode, args.beam_size,
                                       use_amp=True, length_penalty=args.length_penalty)
                for row, text in zip(chunk, texts):
                    text = (text or "").strip()
                    row["clip_text"] = text
                    row["clip_status"] = "ok" if text else "no_text_detected"
            except Exception as e:
                for row in chunk:
                    row["clip_text"], row["clip_status"] = "", f"exception:{e}"
            done = min(i + args.batch_size, len(todo))
            print(f"  [{done}/{len(todo)}]")
            save()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save()
    statuses = {}
    for r in rows:
        statuses[r["clip_status"]] = statuses.get(r["clip_status"], 0) + 1
    print(f"\nDone. Status breakdown: {statuses}")
    print(f"Manifest updated: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
