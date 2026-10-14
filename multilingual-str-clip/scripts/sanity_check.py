"""
sanity_check.py
Debug script for diagnosing poor WRR/CRR on a CLIPSceneTextOCR model. Runs three
independent checks, in order of how likely each is to explain a
catastrophic failure (WRR near 0%, CRR only slightly above noise):

  1. DATA SANITY   — parses your annotation file, reports image aspect
                      ratios (wide/short crops are exactly what the old
                      center-crop preprocessing bug destroyed) and how
                      much of your test-set vocabulary was actually seen
                      during training.
  2. PREPROCESSING  — saves a handful of "what the model actually sees"
     VISUALIZATION    images to disk, so you can eyeball whether text is
                       being cut off, squashed unreadably, etc.
  3. TINY-BATCH      — the most important check: trains on just N samples
     OVERFIT TEST      for many steps and checks the model can memorize
                        them near-perfectly. If it can't, something in the
                        pipeline (data/model/loss/decode) is broken — no
                        amount of extra training data will fix that. If it
                        CAN, the pipeline is sound and poor test accuracy
                        is a data-quantity/training-length/generalization
                        problem instead.

Usage:
    python sanity_check.py \
        --train_ann data/train.tsv \
        --image_root data/images \
        --clip_ckpt /path/to/ViT-B-32.pt \
        --test_ann data/test.tsv \
        --overfit_steps 300 --overfit_n 8
"""
import argparse
import copy
import os
import torch
import torch.nn as nn
from PIL import Image

from vocab import Vocab
from dataset import SceneTextDataset, collate_fn
from model import CLIPSceneTextOCR
from utils import compute_all_metrics, log_device


# --------------------------------------------------------------------------
# 1. DATA SANITY
# --------------------------------------------------------------------------
def check_data(train_ann, image_root, test_ann=None, sample_n=200, vocab_path=None):
    print("\n" + "=" * 60)
    print(" 1. DATA SANITY CHECK")
    print("=" * 60)

    samples = []
    with open(train_ann, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                samples.append(tuple(parts))
    print(f"train_ann samples parsed: {len(samples)}")
    if len(samples) == 0:
        print("  !! No samples parsed — check the TSV separator matches your file "
              "(this script assumes TAB; re-run split_annotations.py if needed).")
        return

    # Aspect ratio check — this is exactly what the old center-crop
    # preprocessing bug destroyed for wide/short word crops.
    ratios = []
    missing_files = 0
    for img_path, _ in samples[:sample_n]:
        full_path = os.path.join(image_root, img_path)
        if not os.path.isfile(full_path):
            missing_files += 1
            continue
        with Image.open(full_path) as im:
            w, h = im.size
            ratios.append(w / max(h, 1))

    if missing_files:
        print(f"  !! {missing_files}/{min(sample_n, len(samples))} sampled image files not found "
              f"under image_root — check --image_root is correct.")
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"  Image width/height ratio over {len(ratios)} sampled images: "
              f"min={min(ratios):.2f} max={max(ratios):.2f} avg={avg_ratio:.2f}")
        if avg_ratio > 1.5:
            print("  -> Images are typically wide/short (normal for word crops). "
                  "Make sure clip_vit.py's preprocessing letterboxes (resize "
                  "preserving aspect ratio + pad) rather than center-cropping, "
                  "or most of each word gets cut off before the model ever "
                  "sees it.")

    # Vocab coverage check. If a vocab_path is given and exists, use THAT
    # exact vocab (the one train.py will actually load) rather than
    # silently rebuilding one from train_ann labels — otherwise this check
    # tells you nothing about whether your real, intended vocab (e.g. a
    # custom hand-provided character list) actually covers your data.
    if vocab_path and os.path.isfile(vocab_path):
        vocab = Vocab.load(vocab_path)
        print(f"  Checking coverage using vocab file: {vocab_path} ({len(vocab.chars)} characters)")
    else:
        vocab = Vocab.build_from_labels([lab for _, lab in samples])
        print(f"  No vocab_path found — checking coverage using a vocab built from train_ann "
              f"labels instead ({len(vocab.chars)} characters). This is NOT necessarily the "
              f"vocab train.py will actually use if --vocab_source or --vocab_path differ.")

    train_chars = set()
    for _, label in samples:
        train_chars.update(label)
    train_unseen = train_chars - set(vocab.chars)
    train_coverage = 1 - len(train_unseen) / max(len(train_chars), 1)
    print(f"  Train-set character coverage by this vocab: {train_coverage*100:.1f}%")
    if train_unseen:
        preview = "".join(sorted(train_unseen)[:30])
        print(f"  {len(train_unseen)} train characters NOT in this vocab (sample): {preview}")
        print("  -> Every occurrence of these in training labels will be encoded as <unk>, "
              "which the model can never map back to the right character at inference time.")

    if test_ann:
        test_chars = set()
        with open(test_ann, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    test_chars.update(parts[1])
        unseen = test_chars - set(vocab.chars)
        coverage = 1 - len(unseen) / max(len(test_chars), 1)
        print(f"  Test-set character coverage by this vocab: {coverage*100:.1f}%")
        if unseen:
            preview = "".join(sorted(unseen)[:30])
            print(f"  {len(unseen)} test characters never seen in training (sample): {preview}")
            print("  -> Any word containing these characters CANNOT be predicted "
                  "correctly no matter how well-trained the model is.")


# --------------------------------------------------------------------------
# 2. PREPROCESSING VISUALIZATION
# --------------------------------------------------------------------------
def dump_preprocessed_samples(preprocess, train_ann, image_root, out_dir, n=6):
    print("\n" + "=" * 60)
    print(" 2. PREPROCESSING VISUALIZATION")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    count = 0
    with open(train_ann, "r", encoding="utf-8") as f:
        for line in f:
            if count >= n:
                break
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            full_path = os.path.join(image_root, img_path)
            if not os.path.isfile(full_path):
                continue
            image = Image.open(full_path).convert("RGB")
            tensor = preprocess(image)
            unnorm = (tensor * std + mean).clamp(0, 1)
            out_img = transforms_to_pil(unnorm)
            out_path = os.path.join(out_dir, f"preproc_{count:02d}_{label}.png")
            out_img.save(out_path)
            print(f"  saved {out_path}  (original size {image.size}, label='{label}')")
            count += 1

    if count == 0:
        print("  No images found to visualize — check --image_root / --train_ann.")
    else:
        print(f"\n  Open the {count} PNGs in {out_dir} and check: is the full word "
              f"visible and legible, or is text cut off / unreadably squashed?")


def transforms_to_pil(tensor):
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(tensor)


# --------------------------------------------------------------------------
# 3/4. TINY-BATCH OVERFIT TEST (frozen backbone, then with backbone unfrozen)
# --------------------------------------------------------------------------
def overfit_tiny_batch(check_num, clip_ckpt, train_ann, image_root, n, steps, lr,
                        device, unfreeze_last_n=0, backbone_lr=3e-5):
    stage_desc = "backbone FROZEN (Stage 1 equivalent)" if unfreeze_last_n == 0 \
        else f"backbone UNFROZEN, last {unfreeze_last_n} block(s) (Stage 2 equivalent)"
    print("\n" + "=" * 60)
    print(f" {check_num}. TINY-BATCH OVERFIT TEST — {stage_desc}")
    print(f"    (n={n} samples, {steps} steps)")
    print("=" * 60)

    samples = []
    with open(train_ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                samples.append(tuple(parts))
    samples = samples[:n]
    if len(samples) < n:
        print(f"  !! Only found {len(samples)} samples, using all of them.")

    vocab = Vocab.build_from_labels([lab for _, lab in samples])
    print(f"  Tiny-batch vocab size: {len(vocab)}")

    model = CLIPSceneTextOCR(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        clip_checkpoint_path=clip_ckpt,
        max_len=32,
        freeze_backbone=True,
        unfreeze_last_n=unfreeze_last_n,
        device=device,
    ).to(device)

    # Build tiny in-memory dataset manually (skip writing a temp TSV file).
    images, token_ids_list, labels = [], [], []
    for img_path, label in samples:
        full_path = os.path.join(image_root, img_path)
        image = Image.open(full_path).convert("RGB")
        images.append(model.preprocess(image))
        token_ids_list.append(torch.tensor(vocab.encode(label), dtype=torch.long))
        labels.append(label)

    max_len = max(len(t) for t in token_ids_list)
    padded = torch.full((len(samples), max_len), vocab.pad_id, dtype=torch.long)
    for i, t in enumerate(token_ids_list):
        padded[i, : len(t)] = t
    images = torch.stack(images).to(device)
    padded = padded.to(device)

    if unfreeze_last_n > 0:
        # Mirror train.py's discriminative LR setup, since a real bug
        # candidate is the backbone destabilizing training the moment it
        # unfreezes if hit with too high an LR.
        backbone_params = list(model.encoder.visual.parameters())
        head_params = (
            list(model.memory_proj.parameters()) + list(model.token_embedding.parameters())
            + list(model.decoder.parameters()) + list(model.output_proj.parameters())
        )
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": backbone_lr}, {"params": head_params, "lr": lr}],
            weight_decay=0.0,  # no regularization for a pure memorization test
        )
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    model.train()
    loss_history = []
    best_loss = float("inf")
    best_state = None
    for step in range(1, steps + 1):
        decoder_input = padded[:, :-1]
        targets = padded[:, 1:]
        logits = model(images, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())

        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"  step {step:4d}/{steps}  loss={loss_val:.4f}")

    # A constant-LR optimizer run past convergence on a tiny fixed batch
    # commonly destabilizes late (Adam's per-parameter moment estimates
    # shrink while loss sits near zero, so a later small gradient produces
    # an oversized step) — that's an artifact of THIS test's fixed-LR
    # setup, not necessarily a bug in the model/pipeline. Report it, but
    # score the best checkpoint seen during training rather than whatever
    # state training happened to end on.
    recent_min = min(loss_history[: len(loss_history) // 2] or loss_history)
    late_avg = sum(loss_history[-5:]) / min(5, len(loss_history))
    if late_avg > recent_min * 1.5:
        print(f"  !! Loss dropped to a minimum of {recent_min:.4f} but rose back up to "
              f"{late_avg:.4f} by the end of training — this is a sign of late-training "
              f"instability (constant LR run past convergence on a tiny fixed batch), not "
              f"necessarily a pipeline bug. Scoring the best checkpoint seen during training "
              f"(loss={best_loss:.4f}) instead of the final, possibly-destabilized one."
              + (f" If this happens repeatedly with --unfreeze_last_n > 0, it may still be "
                 f"worth lowering --backbone_lr in train.py." if unfreeze_last_n > 0 else ""))

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = model.greedy_decode(images, vocab, max_len=32, device=device)
    metrics = compute_all_metrics(preds, labels)

    print("\n  Per-sample results:")
    runaway = 0
    for label, pred in zip(labels, preds):
        mark = "OK " if label == pred else "XX "
        flag = ""
        # len(pred) close to max_len (minus stripped special tokens) is a
        # strong proxy for "never predicted EOS, ran until the hard cutoff".
        never_hit_eos = pred != label and len(pred) >= 30
        much_longer = pred != label and len(pred) > 2 * len(label) + 5
        if never_hit_eos or much_longer:
            flag = "  <- much longer than gt, likely never predicted EOS"
            runaway += 1
        print(f"    [{mark}] gt='{label}'  pred='{pred}'{flag}")

    print(f"\n  Tiny-batch WRR: {metrics['WRR']*100:.1f}%   CRR: {metrics['CRR']*100:.1f}%")

    if metrics["CRR"] < 0.3 and runaway > 0:
        print(f"  !! Even at the best training-loss checkpoint (loss={best_loss:.4f}), CRR is very "
              f"low/negative with {runaway} run-on prediction(s) that never stopped at EOS — this is "
              f"NOT just 'needs more steps' or late-training noise, it's a specific, diagnosable "
              f"issue: the model isn't learning to predict the end-of-sequence token, or teacher-"
              f"forced training loss doesn't transfer to autoregressive decoding (exposure bias). "
              f"Likely causes, in order of likelihood:\n"
              f"       1. EOS position isn't actually being supervised — double check vocab.encode() "
              f"appends eos_id and that decoder_input/targets are shifted by exactly one position in "
              f"both train.py and this script (targets = token_ids[:, 1:]).\n"
              f"       2. ignore_index in the loss is silently excluding real content — confirm "
              f"vocab.pad_id is NOT equal to vocab.eos_id (they must be different token ids).\n"
              f"       3. Too few steps for THIS specific run's 8 samples if they're longer/harder "
              f"than typical — try --overfit_steps 800 --overfit_lr 2e-3 once to rule this out, but "
              f"don't stop investigating if it's still runaway after that.")

    if metrics["WRR"] > 0.8 and metrics["CRR"] > 0.8:
        print("  -> PASS.")
    elif metrics["WRR"] >= 0.5 and metrics["CRR"] >= 0.3:
        print("  -> BORDERLINE: partial memorization — not a clear pass or fail. With real (non-"
              "synthetic) images this can just mean 300 steps / this LR isn't quite enough to fully "
              "converge on 8 samples yet, rather than a structural bug. Before concluding there's a "
              "bug, try --overfit_steps 800 and/or --overfit_lr 2e-3 once; if it still plateaus below "
              "80% WRR, then treat it as a real issue and check the points below.")
    else:
        rerun_note = (
            " Stage 2 (backbone unfrozen) specifically has been observed to fail on an unlucky "
            "random initialization and then pass cleanly on a re-run with no other changes — "
            "if this is Stage 2, re-run this script 1-2 more times before concluding it's a real "
            "bug. A FAIL that reproduces consistently across multiple runs is the one to actually "
            "investigate."
        ) if unfreeze_last_n > 0 else ""
        print("  -> FAIL: the pipeline could not even memorize a handful of "
              "samples after many steps in this configuration. This points to a "
              "real bug in this specific stage — check the preprocessing "
              "visualization above, then double-check the causal/padding mask, "
              "the LR/optimizer setup for this stage, and that images/labels "
              "are actually paired up correctly." + rerun_note)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", required=True)
    ap.add_argument("--image_root", default=None, help="Alias for --train_image_root")
    ap.add_argument("--train_image_root", default=None,
                     help="Folder --train_ann's image paths are relative to (this script only ever "
                          "loads training images — --test_ann is used for label-text vocabulary "
                          "coverage checking only, never for loading test images). Defaults to --image_root.")
    ap.add_argument("--clip_ckpt", required=True)
    ap.add_argument("--test_ann", default=None, help="Optional — enables vocab coverage check")
    ap.add_argument("--vocab_path", default="telugu_vocab.json",
                     help="Checks coverage against THIS vocab file if it exists (matching train.py's "
                          "default) — otherwise falls back to a vocab built from --train_ann labels.")
    ap.add_argument("--preproc_out_dir", default="sanity_preproc_samples")
    ap.add_argument("--overfit_n", type=int, default=8)
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--overfit_lr", type=float, default=1e-3)
    ap.add_argument("--overfit_backbone_lr", type=float, default=3e-5,
                     help="Should match train.py's --backbone_lr for the Stage-2 check to be meaningful")
    ap.add_argument("--overfit_unfreeze_last_n", type=int, default=2,
                     help="Should match train.py's --unfreeze_last_n for the Stage-2 check to be meaningful")
    ap.add_argument("--skip_overfit_test", action="store_true")
    ap.add_argument("--skip_stage2_test", action="store_true",
                     help="Skip the backbone-unfrozen overfit check (only run the Stage-1/frozen one)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_image_root = args.train_image_root or args.image_root
    if not train_image_root:
        raise ValueError("Provide --train_image_root (or --image_root as an alias).")

    device = torch.device(args.device)
    log_device(device, "sanity_check.py")

    check_data(args.train_ann, train_image_root, args.test_ann, vocab_path=args.vocab_path)

    # Build preprocess from the checkpoint (doesn't need the full OCR model).
    from clip_vit import load_clip_visual
    _, preprocess = load_clip_visual(args.clip_ckpt, device="cpu")
    dump_preprocessed_samples(preprocess, args.train_ann, train_image_root, args.preproc_out_dir)

    if not args.skip_overfit_test:
        stage1 = overfit_tiny_batch(
            3, args.clip_ckpt, args.train_ann, train_image_root,
            args.overfit_n, args.overfit_steps, args.overfit_lr, device,
            unfreeze_last_n=0,
        )
        stage2 = None
        if not args.skip_stage2_test:
            stage2 = overfit_tiny_batch(
                4, args.clip_ckpt, args.train_ann, train_image_root,
                args.overfit_n, args.overfit_steps, args.overfit_lr, device,
                unfreeze_last_n=args.overfit_unfreeze_last_n, backbone_lr=args.overfit_backbone_lr,
            )

        print("\n" + "=" * 60)
        print(" OVERALL VERDICT")
        print("=" * 60)
        if stage1["WRR"] <= 0.8:
            print("  Stage 1 (frozen backbone) FAILED to memorize — the core pipeline "
                  "(data/model/loss/decode) has a bug independent of the two-stage schedule.")
        elif stage2 is not None and stage2["WRR"] <= 0.8:
            print("  Stage 1 passed but Stage 2 (backbone unfrozen) FAILED — the bug is "
                  "specific to backbone fine-tuning, most likely --backbone_lr too high, "
                  "no LR warmup at the unfreeze point, or --unfreeze_last_n too large for "
                  "your dataset size. This can fully explain good tiny-batch / poor "
                  "full-dataset results even though 'the pipeline works'. NOTE: Stage 2 has "
                  "been observed to fail on an unlucky random init and pass cleanly on a "
                  "re-run with no other changes — re-run this script 1-2 more times before "
                  "concluding it's a real, reproducible bug.")
        else:
            print("  Both stages passed. The pipeline itself (including backbone "
                  "fine-tuning) is sound. Poor full-dataset accuracy is a "
                  "data-quantity / training-length / augmentation / hyperparameter "
                  "issue — see the accuracy tips in README.md (class-weighted loss "
                  "for long-tail conjuncts, augmentation, beam search, more epochs, "
                  "more data).")

    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
