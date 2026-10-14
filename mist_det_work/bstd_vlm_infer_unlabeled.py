#!/usr/bin/env python3
"""
bstd_vlm_infer_unlabeled.py
Run a bstd_vlm_lora_train.py LoRA adapter (Qwen3-VL) on a folder of
UNLABELED crop images -- no test_tsv/ground truth required. Same beam
search + optional lexicon rerank as bstd_vlm_beam_rerank_eval.py, minus
the scoring/eval loop, for use on new images (e.g. MIST crops) with no
transcription ground truth. The rerank()/build_lexicon()/align()/
nearest_neighbor_correct() functions below are copied verbatim from
bstd_vlm_beam_rerank_eval.py (itself copied verbatim from
bstd_lexicon_rerank_eval.py) -- see that file's module docstring for the
full regression history behind each fix. Do not edit these independently;
if a future fix is validated there, port it here too.

Usage (Kannada, best single checkpoint):
  python3 bstd_vlm_infer_unlabeled.py \\
      --adapter_dir checkpoints/kannada_qwen3vl_lora_visiontuned_augmented_v2 \\
      --image_dir ~/mist_det_work/recognition_crops/kannada \\
      --lexicon_ann data/kannada/train/train.tsv data/kannada/val/val.tsv vocab/kannada_indiccorp_lexicon.tsv \\
      --beam_size 5 --length_penalty 1.0 \\
      --output_csv ~/mist_det_work/recognition_crops/kannada_predictions.csv

Omit --lexicon_ann to skip reranking and just report raw top-1 beam output.

Needs: pip install --break-system-packages transformers accelerate peft bitsandbytes pillow
(same as bstd_vlm_beam_rerank_eval.py -- already installed if you've run that before).
"""
import argparse
import csv
import os
import unicodedata
from collections import Counter, defaultdict

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

PROMPT = "Read the text written in this image. Respond with only the exact text, nothing else."
TRAILING_PUNCT = ".,;:!?-"
# Telugu-specific, inert for Kannada -- kept for byte-for-byte parity with
# the source files this was copied from (see module docstring).
ANUSVARA_VARIANT_NORMALIZE = str.maketrans({"౦": "ం"})


def normalize_anusvara_variant(s):
    return s.translate(ANUSVARA_VARIANT_NORMALIZE)


# ---- verbatim from bstd_lexicon_rerank_eval.py / bstd_vlm_beam_rerank_eval.py ----
def build_lexicon(ann_paths):
    word_freq = Counter()
    for path in ann_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[1]:
                    word_freq[parts[1]] += 1
    return set(word_freq.keys()), word_freq


def align(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i; bt[i][0] = "D"
    for j in range(n + 1):
        dp[0][j] = j; bt[0][j] = "I"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]; bt[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                bt[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    edit_positions_from_end = []
    i, j = m, n
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1; j -= 1
        elif op == "S":
            edit_positions_from_end.append(n - j); i -= 1; j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            edit_positions_from_end.append(n - j); j -= 1
    return dp[m][n], edit_positions_from_end


def nearest_neighbor_correct(prediction, lexicon, by_length, max_edit_distance, suffix_guard_chars,
                              word_freq=None, tie_break_by_frequency=True):
    if prediction in lexicon or not prediction:
        return prediction, False, "n/a"
    candidates = []
    for length in range(max(1, len(prediction) - max_edit_distance),
                         len(prediction) + max_edit_distance + 1):
        candidates.extend(by_length.get(length, []))
    scored = []
    for word in candidates:
        d, positions = align(prediction, word)
        if d <= max_edit_distance:
            scored.append((d, positions, word))
    if not scored:
        return prediction, False, "no_close_match"
    best_dist = min(s[0] for s in scored)
    best = [s for s in scored if s[0] == best_dist]
    if len(best) > 1:
        if tie_break_by_frequency and word_freq:
            freqs = [word_freq.get(w, 0) for _, _, w in best]
            max_freq = max(freqs)
            top_by_freq = [b for b, f in zip(best, freqs) if f == max_freq]
            if len(top_by_freq) == 1:
                _, positions, word = top_by_freq[0]
                if suffix_guard_chars > 0 and positions and all(p < suffix_guard_chars for p in positions):
                    return prediction, False, "declined_suffix_guard"
                return word, True, f"corrected_dist_{best_dist}_freq_tiebreak"
        return prediction, False, f"ambiguous_{len(best)}_tied"
    _, positions, word = best[0]
    if suffix_guard_chars > 0 and positions and all(p < suffix_guard_chars for p in positions):
        return prediction, False, "declined_suffix_guard"
    return word, True, f"corrected_dist_{best_dist}"


def rerank(candidates, lexicon, by_length, max_edit_distance, suffix_guard_chars,
           word_freq=None, tie_break_by_frequency=True, max_length_ratio=0.5):
    top1_text = candidates[0][0]
    top1_len = len(top1_text)
    if " " in top1_text:
        return top1_text, "top1_multiword_kept_unreranked"
    top1_stripped = top1_text.rstrip(TRAILING_PUNCT)
    if top1_text in lexicon:
        return top1_text, "top1_already_valid"
    if top1_stripped and top1_stripped != top1_text and top1_stripped in lexicon:
        return top1_text, "top1_already_valid_mod_punct"
    top1_anusvara_norm = normalize_anusvara_variant(top1_text)
    if top1_anusvara_norm != top1_text and top1_anusvara_norm in lexicon:
        return top1_text, "top1_already_valid_mod_anusvara_variant"
    comma_period_swapped = top1_text.translate(str.maketrans(",.", ".,"))
    if comma_period_swapped != top1_text and comma_period_swapped in lexicon:
        return comma_period_swapped, "corrected_comma_period_swap"
    for rank, (text, score) in enumerate(candidates[1:], start=1):
        if text in lexicon:
            if top1_len == 0 or abs(len(text) - top1_len) / top1_len <= max_length_ratio:
                return text, f"beam_alternative_rank_{rank}"
    corrected, was_corrected, reason = nearest_neighbor_correct(
        top1_text, lexicon, by_length, max_edit_distance, suffix_guard_chars,
        word_freq=word_freq, tie_break_by_frequency=tie_break_by_frequency)
    if was_corrected:
        return corrected, f"nearest_neighbor_{reason}"
    return top1_text, f"kept_top1_{reason}"
# ---- end verbatim block ----


@torch.no_grad()
def get_beam_candidates(model, processor, image, device, beam_size, length_penalty, max_new_tokens=32):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, num_beams=beam_size,
        num_return_sequences=beam_size, do_sample=False, early_stopping=True,
        length_penalty=length_penalty, output_scores=True, return_dict_in_generate=True,
    )
    sequences = output.sequences
    scores = output.sequences_scores
    candidates = []
    for seq, score in zip(sequences, scores):
        text = processor.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        text = unicodedata.normalize("NFC", text)
        candidates.append((text, score.item()))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--lexicon_ann", nargs="*", default=[],
                     help="Optional TSV lexicon source(s) for rerank correction. Omit to skip reranking.")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_edit_distance", type=int, default=1)
    ap.add_argument("--suffix_guard_chars", type=int, default=1)
    ap.add_argument("--max_length_ratio", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f"!! --image_dir not found: {args.image_dir}")

    lexicon, word_freq, by_length = set(), None, defaultdict(list)
    if args.lexicon_ann:
        print(f"Building lexicon from: {', '.join(args.lexicon_ann)}")
        lexicon, word_freq = build_lexicon(args.lexicon_ann)
        for w in lexicon:
            by_length[len(w)].append(w)
        print(f"Lexicon size: {len(lexicon)} unique word(s)")
    else:
        print("No --lexicon_ann given -- skipping lexicon rerank, reporting raw top-1 beam output.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! WARNING: no CUDA device found -- 4-bit inference requires a GPU.")

    print(f"Loading processor + base model {args.base_model_id} ...")
    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True,
    )
    print(f"Attaching LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    fnames = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(exts))
    if not fnames:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"Found {len(fnames)} image(s) in {args.image_dir}.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "top1_beam", "final", "method", "top1_score"])
        for i, fname in enumerate(fnames):
            path = os.path.join(args.image_dir, fname)
            image = Image.open(path).convert("RGB")
            candidates = get_beam_candidates(model, processor, image, device,
                                              beam_size=args.beam_size, length_penalty=args.length_penalty,
                                              max_new_tokens=args.max_new_tokens)
            top1_text, top1_score = candidates[0]
            if lexicon:
                final, method = rerank(candidates, lexicon, by_length, args.max_edit_distance,
                                        args.suffix_guard_chars, word_freq=word_freq,
                                        tie_break_by_frequency=True, max_length_ratio=args.max_length_ratio)
            else:
                final, method = top1_text, "no_rerank"
            print(f"[{i+1}/{len(fnames)}] {fname}\t{final}")
            writer.writerow([fname, top1_text, final, method, top1_score])

    print(f"\nSaved {len(fnames)} prediction(s) to: {args.output_csv}")


if __name__ == "__main__":
    main()
