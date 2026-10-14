#!/usr/bin/env python3
"""
phase1_vlm_lora_train.py -- QLoRA fine-tuning of Qwen3-VL-2B-Instruct (or any
compatible Qwen-VL checkpoint) on Phase 1 / IndicSTR12 scene-text crops, one
language at a time. This is a direct generalization of the BSTD project's
bstd_vlm_lora_train.py (which produced the validated Kannada-BSTD VLM LoRA
results: 46.52% WRR final, v2+v1 ensemble) -- the training logic is UNCHANGED
from that script; only the docstring/example paths and the dataset class name
have been generalized away from BSTD-Kannada specifics, since --train_tsv/
--image_root/--output_dir were already fully generic CLI arguments there.

WHY THIS EXISTS (Phase 1 VLM fine-tuning, per the IJDAR paper's Future Work)
  The IJDAR paper's MIST recognition benchmark found Qwen3-VL-2B zero-shot
  scores only 6.80% WRR on MIST vs. Track1-CLIP's 39.34% -- far behind, and
  BSTD's own Kannada precedent showed LoRA fine-tuning closes a similar gap
  substantially (zero-shot ~30% WRR -> LoRA fine-tuned 46.52% WRR). This
  script runs the same recipe on Phase 1's OWN training data (the officially
  released IndicSTR12 real train split, data/<lang>/train/train.tsv) for
  each of the 10 languages Phase 1 already covers, producing a VLM checkpoint
  comparable to each language's existing CLIP ViT-B/16 checkpoint on the
  same IndicSTR12 test split (Table 1 of the paper) -- then, in a later
  step, those same checkpoints get applied to MIST's crops as a "Track 2"
  VLM number, the same way Track1-CLIP transferred unmodified.

WHAT THIS DOES
  - Loads the base VLM in 4-bit (bitsandbytes NF4) to fit a 7.6GB-VRAM GPU.
  - By default, attaches LoRA adapters to the TEXT decoder's attention/MLP
    projections only (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
    down_proj) -- vision tower frozen by construction.
  - Pass --tune_vision to ALSO LoRA-adapt the vision tower's attention
    (qkv + proj) in every block. RECOMMENDED DEFAULT for Phase 1 runs: the
    BSTD Kannada precedent found frozen-vision training produces a healthy
    loss curve but fluent, image-UNGROUNDED predictions (WRR near 0) --
    diagnosed as the LoRA-tuned text decoder learning "what <language> words
    look like" from the label distribution without adapting the vision
    tower's weak out-of-domain glyph signal. --tune_vision fixed this for
    BSTD Kannada and is the recipe this project's driver script
    (train_all_phase1_vlm.sh) uses for every language by default, skipping
    the known-bad frozen-vision baseline to save GPU time across 10
    languages. Run frozen-vision manually first for any language if you
    want to re-confirm the failure mode before trusting this default.
  - Frames each (image, label) pair as a single-turn OCR instruction:
    user turn = image + "read the text" prompt, assistant turn = the label.
  - Masks the loss to the assistant's tokens only (completion-only SFT) via
    a longest-common-prefix comparison between the prompt-only and full
    encodings.
  - Normalizes every label to Unicode NFC before use (guards against the
    same legacy-decomposed-vowel-sign encoding quirk found in BSTD Kannada
    labels and, separately, in this project's own Telugu/Kannada GT digit
    encoding bugs).

IMPORTANT CAVEATS (read before running)
  - Needs: pip install --break-system-packages transformers accelerate
    peft bitsandbytes pillow -- same environment bstd_vlm_lora_train.py
    already ran successfully in on this machine.
  - Needs an NVIDIA GPU (your RTX 4060, 7.6GB VRAM); no CPU fallback.
  - If you hit an OOM: lower --per_device_train_batch_size (already
    defaults to 1), raise --gradient_accumulation_steps to compensate, or
    lower --max_pixels. ALSO set this env var when using --tune_vision
    (must be visible before CUDA initializes, not a script flag):
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 phase1_vlm_lora_train.py ...
    A --tune_vision run WITHOUT this env var OOM'd from allocator
    fragmentation (not genuinely out of memory) on this exact GPU during
    the BSTD Kannada work -- do not skip it.
  - dataloader_num_workers defaults to 0 deliberately (see BSTD precedent).
  - --save_steps/--resume exist so a crash loses at most --save_steps
    steps, not the whole run -- use --resume after any interruption.

Usage (smoke test first, ALWAYS, before the full run):
  python3 phase1_vlm_lora_train.py \\
      --train_tsv data/telugu/train/train.tsv \\
      --image_root data/telugu/train/images \\
      --output_dir checkpoints/telugu_qwen3vl_lora_visiontuned_smoketest \\
      --tune_vision --max_pixels 200704 \\
      --max_train_samples 8 --num_train_epochs 1

Full run (after the smoke test passes cleanly) -- note the required env var:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 phase1_vlm_lora_train.py \\
      --train_tsv data/telugu/train/train.tsv \\
      --image_root data/telugu/train/images \\
      --output_dir checkpoints/telugu_qwen3vl_lora_visiontuned \\
      --tune_vision --max_pixels 200704

Resuming after a crash -- same command, same --output_dir, plus --resume:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 phase1_vlm_lora_train.py \\
      --train_tsv data/telugu/train/train.tsv \\
      --image_root data/telugu/train/images \\
      --output_dir checkpoints/telugu_qwen3vl_lora_visiontuned \\
      --tune_vision --max_pixels 200704 --resume

In practice, use train_all_phase1_vlm.sh to run all 10 languages back to
back with resume-safety and per-language completion markers, rather than
invoking this script by hand 10 times.
"""
import argparse
import os
import random
import unicodedata

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROMPT = "Read the text written in this image. Respond with only the exact text, nothing else."

TEXT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]  # text-decoder-only by construction: Qwen3-VL's vision tower uses
   # "qkv"/"proj" module names instead, so these never match it -- vision
   # stays frozen without extra code.

# Single regex (PEFT applies this via re.fullmatch against each module's
# full dotted name when target_modules is a string) covering the text
# decoder's projections AND the vision tower's attention qkv/proj -- used
# only when --tune_vision is passed. Verified (in the original BSTD work)
# against the actual Qwen3-VL module tree: matches vision attention's
# "...attn.qkv" / "...attn.proj" but NOT patch_embed's "...patch_embed.proj"
# (a Conv3d, not a Linear) and NOT the merger's "linear_fc1"/"linear_fc2".
TEXT_AND_VISION_LORA_TARGET_REGEX = (
    r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    r"|.*attn\.(qkv|proj)$"
)


def load_rows(train_tsv, image_root, max_samples=None, seed=42):
    rows = []
    with open(train_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1]:
                rows.append((parts[0], parts[1]))
    random.Random(seed).shuffle(rows)
    samples = []
    for rel_path, label in rows:
        full_path = rel_path if os.path.isfile(rel_path) else os.path.join(image_root, os.path.basename(rel_path))
        if not os.path.isfile(full_path):
            continue
        samples.append((full_path, unicodedata.normalize("NFC", label)))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


class OCRDataset(Dataset):
    """Yields per-sample encoded tensors, with the assistant-only loss mask
    already computed. Encoding happens here (not in a batched collate_fn)
    because the prompt/full token-length boundary must be computed against
    each sample's OWN image (the image-placeholder token count depends on
    that image's resolution), which is only correct per-sample, not
    batched."""

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        user_turn = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        full_turn = user_turn + [{"role": "assistant", "content": [{"type": "text", "text": label}]}]

        prompt_text = self.processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_turn, tokenize=False, add_generation_prompt=False)

        prompt_enc = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_enc = self.processor(text=[full_text], images=[image], return_tensors="pt")

        prompt_ids = prompt_enc["input_ids"][0]
        full_ids = full_enc["input_ids"][0]

        # Robust mask boundary: longest common prefix of the two encodings,
        # rather than blindly trusting len(prompt_ids) -- guards against a
        # rare BPE merge-boundary shift right at the prompt/response join.
        boundary = 0
        for a, b in zip(prompt_ids.tolist(), full_ids.tolist()):
            if a != b:
                break
            boundary += 1

        labels = full_ids.clone()
        labels[:boundary] = -100

        item = {
            "input_ids": full_ids,
            "attention_mask": full_enc["attention_mask"][0],
            "labels": labels,
            "pixel_values": full_enc["pixel_values"],
        }
        if "image_grid_thw" in full_enc:
            item["image_grid_thw"] = full_enc["image_grid_thw"]
        # Qwen3-VL's forward() hard-requires this alongside image_grid_thw to
        # compute multimodal (M-RoPE) position ids -- thread it through like
        # every other processor output instead of silently dropping it.
        if "mm_token_type_ids" in full_enc:
            item["mm_token_type_ids"] = full_enc["mm_token_type_ids"][0]
        return item


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)

        def pad_1d(t, value):
            pad_len = max_len - t.shape[0]
            if pad_len == 0:
                return t
            return torch.cat([t, torch.full((pad_len,), value, dtype=t.dtype)], dim=0)

        input_ids = torch.stack([pad_1d(item["input_ids"], pad_token_id) for item in batch])
        attention_mask = torch.stack([pad_1d(item["attention_mask"], 0) for item in batch])
        labels = torch.stack([pad_1d(item["labels"], -100) for item in batch])
        # Each sample's pixel_values is (num_patches_i, patch_dim) -- Qwen-VL
        # expects all images' patches concatenated along dim 0 for a batch,
        # with image_grid_thw (one row per image) telling it how to split
        # them back apart internally.
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = torch.stack(
                [pad_1d(item["mm_token_type_ids"], 0) for item in batch]
            )
        return out

    return collate_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_train_samples", type=int, default=None, help="Smoke-test with e.g. 8 before a full run.")
    ap.add_argument("--num_train_epochs", type=float, default=1.0,
                     help="Default 1 (not 3) -- Phase 1's real-data train splits are ~9-10x larger than BSTD "
                          "Kannada's, and this run covers 10 languages sequentially. Raise for any language "
                          "individually after seeing its first-epoch eval number.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--tune_vision", action="store_true",
                     help="Also LoRA-adapt the vision tower's attention (qkv+proj) in every block, "
                          "instead of leaving it fully frozen. See module docstring -- recommended default "
                          "for Phase 1 runs based on the BSTD Kannada precedent.")
    ap.add_argument("--max_pixels", type=int, default=None,
                     help="Optional cap on image_processor.max_pixels to shrink image-token count "
                          "per sample if you hit OOM (Qwen-VL uses dynamic resolution).")
    ap.add_argument("--dataloader_num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_steps", type=int, default=300,
                     help="Save a checkpoint every N optimizer steps (in addition to end-of-epoch), "
                          "so a crash loses at most --save_steps steps of progress.")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from the latest checkpoint under --output_dir instead of training "
                          "from scratch.")
    args = ap.parse_args()

    print(f"Loading processor for {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if args.max_pixels:
        processor.image_processor.max_pixels = args.max_pixels
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {args.model_id} in 4-bit ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = TEXT_AND_VISION_LORA_TARGET_REGEX if args.tune_vision else TEXT_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_trainable_vision = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "visual" in n
    )
    n_trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = "should be > 0 (--tune_vision was passed)" if args.tune_vision else "should be 0"
    print(f"Trainable params in vision tower: {n_trainable_vision} ({expected}). "
          f"Total trainable: {n_trainable_total}.")
    if args.tune_vision and n_trainable_vision == 0:
        print("!! WARNING: --tune_vision was passed but 0 vision params are trainable -- the regex may "
              "not be matching this checkpoint's actual module names. Double-check before trusting a full run.")
    elif not args.tune_vision and n_trainable_vision != 0:
        print("!! WARNING: expected 0 trainable vision-tower params -- target_modules may be matching "
              "vision module names on this checkpoint. Double-check before trusting a full run.")

    print(f"Loading training rows from {args.train_tsv} ...")
    samples = load_rows(args.train_tsv, args.image_root, max_samples=args.max_train_samples, seed=args.seed)
    print(f"Training on {len(samples)} samples"
          + (f" (smoke test, --max_train_samples={args.max_train_samples})" if args.max_train_samples else "") + ".")
    if len(samples) < 4:
        raise SystemExit(f"Only found {len(samples)} valid (image, label) pairs -- check --train_tsv/--image_root.")

    dataset = OCRDataset(samples, processor)
    collate_fn = make_collate_fn(processor.tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    if args.resume:
        print(f"Resuming from the latest checkpoint under {args.output_dir} ...")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"Saving LoRA adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
