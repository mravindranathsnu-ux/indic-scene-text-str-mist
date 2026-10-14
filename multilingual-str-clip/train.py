"""
train.py
Fine-tunes CLIPSceneTextOCR on a labeled scene-text dataset, for Telugu,
another Indian script, or English — pass --language (see vocab.py's
UNICODE_BLOCKS for supported values). One model = one language; train a
separate model per language/script rather than one combined multilingual
model.
Single-GPU oriented: mixed precision (AMP), gradient accumulation,
LR warmup/cosine decay, discriminative LR + class-weighted loss for
long-tail/conjunct characters, and training-time augmentation.

Example (Telugu, backward-compatible defaults):
    python train.py \
        --train_ann data/telugu/train.tsv --val_ann data/telugu/val.tsv \
        --image_root data/telugu/images --clip_ckpt ./weights/ViT-B-32.pt \
        --epochs 40 --batch_size 32

Example (Hindi):
    python train.py --language hindi \
        --train_ann data/hindi/train.tsv --val_ann data/hindi/val.tsv \
        --image_root data/hindi/images --clip_ckpt ./weights/ViT-B-32.pt \
        --vocab_path hindi_vocab.json --out_dir checkpoints/hindi \
        --epochs 40 --batch_size 32
"""
import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vocab import Vocab
from dataset import SceneTextDataset, collate_fn
from model import CLIPSceneTextOCR
from utils import compute_cer, log_device


def raw_state_dict(model):
    """torch.compile() wraps the model and prefixes state_dict keys with
    '_orig_mod.', which would silently break loading in inference.py /
    evaluate.py (they build a plain CLIPSceneTextOCR and load_state_dict
    directly). Always save the underlying module's weights instead."""
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def build_vocab(args):
    if args.vocab_path and os.path.exists(args.vocab_path):
        vocab = Vocab.load(args.vocab_path)
        print(f"Vocab: loaded {len(vocab.chars)} characters from existing {args.vocab_path} "
              f"(--vocab_source={args.vocab_source} was IGNORED — a vocab_path file on disk "
              f"always takes priority over --vocab_source; delete/rename the file first if you "
              f"actually want to rebuild the vocab a different way).")
        return vocab

    if args.vocab_source == "unicode_block":
        # Covers the entire Unicode block for --language (default: telugu)
        # plus common punctuation/digits up front, regardless of what
        # happens to appear in --train_ann. Guarantees no "unseen
        # character" cliff at test time.
        vocab = Vocab.build_from_unicode_block(args.language)
    else:
        # Build only from characters actually seen in --train_ann.
        labels = []
        with open(args.train_ann, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    labels.append("\t".join(parts[1:]))
        vocab = Vocab.build_from_labels(labels)

    print(f"Vocab: built {len(vocab.chars)} characters from --vocab_source={args.vocab_source} "
          f"(language={args.language}).")
    if args.vocab_path:
        vocab.save(args.vocab_path)
        print(f"       Saved to {args.vocab_path} (future runs will load this file directly "
              f"unless you delete it — see note above).")
    return vocab


def compute_class_weights(vocab, train_ann):
    """
    Inverse-frequency class weights for the loss, to help with long-tail
    characters (e.g. Telugu/Indic conjuncts) — common vowels/consonants show
    up in nearly every word, while many rarer glyphs appear infrequently, so
    an unweighted loss mostly just learns the common ones well and
    under-trains the rare-but-important conjuncts. sqrt-damping keeps this
    from swinging too far the other way (extremely rare noise characters
    getting huge weight).
    """
    counts = torch.ones(len(vocab))  # +1 smoothing avoids div-by-zero
    with open(train_ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            for ch in parts[1]:
                idx = vocab.stoi.get(ch)
                if idx is not None:
                    counts[idx] += 1

    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.mean()  # keep overall loss scale roughly unchanged
    weights[vocab.pad_id] = 1.0          # irrelevant (ignore_index excludes it), kept tidy
    return weights


def make_lr_lambda(warmup_steps, total_steps, start_step=0, min_lr_frac=0.05):
    """Linear warmup then cosine decay, optionally starting partway through
    training (used to warm the backbone's LR back up right when it
    unfreezes in Stage 2, instead of hitting it with the full LR
    instantly — a likely source of training instability otherwise)."""
    def lr_lambda(step):
        s = step - start_step
        if s < 0:
            return 0.0
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        span = max(1, total_steps - start_step - warmup_steps)
        progress = min(max((s - warmup_steps) / span, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_frac + (1 - min_lr_frac) * cosine
    return lr_lambda


def run_epoch(model, loader, optimizer, criterion, device, scaler, scheduler,
              grad_accum_steps, use_amp, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    if train:
        optimizer.zero_grad()

    context = torch.enable_grad() if train else torch.no_grad()
    n = len(loader)
    with context:
        for i, batch in enumerate(tqdm(loader, desc="train" if train else "val")):
            images = batch["images"].to(device, non_blocking=True)
            token_ids = batch["token_ids"].to(device, non_blocking=True)

            decoder_input = token_ids[:, :-1]
            targets = token_ids[:, 1:]

            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(images, decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if train:
                scaler.scale(loss / grad_accum_steps).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == n:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_cer(model, loader, vocab, device, max_len, use_amp=True):
    model.eval()
    preds, refs = [], []
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            decoded = model.greedy_decode(images, vocab, max_len=max_len, device=device)
            preds.extend(decoded)
            refs.extend(batch["labels"])
    return compute_cer(preds, refs)


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def print_all_params(args):
    """Prints every training parameter, grouped for readability, before
    anything expensive (model/dataset loading) happens."""
    groups = {
        "Data": ["train_ann", "val_ann", "image_root", "train_image_root", "val_image_root"],
        "Language / vocab": ["language", "vocab_path", "vocab_source"],
        "Model": ["clip_ckpt", "max_len"],
        "Two-stage fine-tuning": ["warmup_frozen_epochs", "unfreeze_last_n", "lr", "backbone_lr",
                                   "head_warmup_steps", "backbone_warmup_steps", "min_lr_frac"],
        "Regularization / data augmentation": ["augment", "class_weights"],
        "Training loop": ["batch_size", "grad_accum_steps", "epochs", "patience", "resume"],
        "Single-GPU efficiency": ["amp", "compile_model", "num_workers"],
        "Output": ["out_dir", "device"],
    }
    args_dict = vars(args)
    seen = set()
    print("\n" + "=" * 60)
    print(" TRAINING PARAMETERS")
    print("=" * 60)
    for group_name, keys in groups.items():
        present = [k for k in keys if k in args_dict]
        if not present:
            continue
        print(f"\n[{group_name}]")
        for k in present:
            print(f"  --{k:<22s} {args_dict[k]}")
            seen.add(k)
    leftover = [k for k in args_dict if k not in seen]
    if leftover:
        print(f"\n[Other]")
        for k in sorted(leftover):
            print(f"  --{k:<22s} {args_dict[k]}")
    print("=" * 60 + "\n")


def confirm_or_exit(args):
    if args.yes:
        print("(--yes given, skipping confirmation prompt)\n")
        return
    try:
        answer = input("Proceed with training using the parameters above? [y/N]: ").strip().lower()
    except EOFError:
        # No interactive terminal attached (e.g. piped/non-interactive CI run) --
        # fail safe by refusing to silently start training with unconfirmed params.
        print("\nNo interactive input available — re-run with --yes to skip confirmation "
              "in non-interactive contexts.")
        raise SystemExit(1)
    if answer not in ("y", "yes"):
        print("Aborted — no training started.")
        raise SystemExit(0)
    print()


def write_summary_file(path, args, num_train, num_val, elapsed_seconds, epoch_history,
                        best_cer, best_epoch, stopped_early):
    lines = []
    lines.append("=" * 60)
    lines.append(" TRAINING SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Language           : {args.language}")
    lines.append(f"Output dir          : {args.out_dir}")
    lines.append(f"Training samples     : {num_train}")
    lines.append(f"Validation samples    : {num_val}")
    lines.append(f"Total wall-clock time  : {format_duration(elapsed_seconds)}")
    lines.append(f"Epochs run              : {len(epoch_history)} (configured max: {args.epochs})")
    lines.append(f"Stopped early           : {stopped_early}")
    if epoch_history:
        lines.append(f"Best val_CER             : {best_cer:.4f}  (epoch {best_epoch})")
        lines.append(f"First epoch train_loss     : {epoch_history[0]['train_loss']:.4f}")
        lines.append(f"Last epoch train_loss        : {epoch_history[-1]['train_loss']:.4f}")
        lines.append(f"Min train_loss                : {min(h['train_loss'] for h in epoch_history):.4f}")
        lines.append(f"Min val_loss                    : {min(h['val_loss'] for h in epoch_history):.4f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(" PARAMETERS")
    lines.append("-" * 60)
    for k, v in sorted(vars(args).items()):
        lines.append(f"  --{k:<22s} {v}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(" PER-EPOCH HISTORY")
    lines.append("-" * 60)
    lines.append(f"{'epoch':>6s} {'train_loss':>12s} {'val_loss':>10s} {'val_CER':>9s} "
                  f"{'lr_backbone':>12s} {'lr_head':>10s}")
    for h in epoch_history:
        lines.append(f"{h['epoch']:6d} {h['train_loss']:12.4f} {h['val_loss']:10.4f} "
                      f"{h['val_cer']:9.4f} {h['lr_backbone']:12.2e} {h['lr_head']:10.2e}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Also write a machine-readable copy alongside, for later programmatic
    # comparison across runs/languages.
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "params": {k: v for k, v in vars(args).items()},
            "num_train_samples": num_train,
            "num_val_samples": num_val,
            "elapsed_seconds": elapsed_seconds,
            "epochs_run": len(epoch_history),
            "stopped_early": stopped_early,
            "best_cer": best_cer,
            "best_epoch": best_epoch,
            "epoch_history": epoch_history,
        }, f, indent=2, ensure_ascii=False)

    return json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", required=True)
    ap.add_argument("--val_ann", required=True)
    ap.add_argument("--image_root", default=None,
                     help="Shared image root, used for BOTH train and val if --train_image_root / "
                          "--val_image_root aren't given. Kept for backward compatibility with a "
                          "single-folder dataset layout — for separate train/val image sets "
                          "(e.g. a database that ships train/val/test as distinct folders), use "
                          "--train_image_root / --val_image_root instead.")
    ap.add_argument("--train_image_root", default=None,
                     help="Folder --train_ann's image paths are relative to. Defaults to --image_root.")
    ap.add_argument("--val_image_root", default=None,
                     help="Folder --val_ann's image paths are relative to. Defaults to --image_root.")
    ap.add_argument("--language", default="telugu",
                     help="Script/language for --vocab_source=unicode_block (see vocab.py's "
                          "UNICODE_BLOCKS for supported values: telugu, hindi/devanagari, tamil, "
                          "malayalam, kannada, bengali, gujarati, gurmukhi/punjabi, odia, "
                          "english/latin). Only affects vocab building — has no effect if "
                          "--vocab_path already exists on disk, or with --vocab_source=train_labels. "
                          "Default 'telugu' preserves this project's original behavior.")
    ap.add_argument("--vocab_path", default="telugu_vocab.json",
                     help="Default is Telugu-specific for backward compatibility — for other "
                          "languages, pass e.g. --vocab_path hindi_vocab.json explicitly so you "
                          "don't accidentally load/overwrite a different language's vocab file.")
    ap.add_argument("--vocab_source", choices=["unicode_block", "train_labels"], default="unicode_block",
                     help="'unicode_block' (default) covers the full --language Unicode block up front; "
                          "'train_labels' builds a smaller vocab from only the characters seen in --train_ann")
    ap.add_argument("--clip_ckpt", required=True,
                     help="Path to local CLIP checkpoint file, e.g. ./weights/ViT-B-32.pt or ViT-B-16.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                     help="Accumulate gradients over this many batches before each optimizer step — "
                          "use to simulate a larger effective batch size on a single GPU with limited memory")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4,
                     help="Peak LR for the new (randomly-initialized) decoder + projection + embeddings")
    ap.add_argument("--backbone_lr", type=float, default=3e-5,
                     help="Peak LR for the pretrained CLIP backbone once it starts unfreezing")
    ap.add_argument("--head_warmup_steps", type=int, default=None,
                     help="Linear LR warmup steps for the decoder head at the very start of training. "
                          "Default: 10%% of total training steps (min 20) — set explicitly to override. "
                          "IMPORTANT: a fixed large value (e.g. 500) can silently eat most/all of a small "
                          "dataset's training run, leaving the LR near-zero the whole time — this is a "
                          "common cause of val_CER staying stuck near its worst value.")
    ap.add_argument("--backbone_warmup_steps", type=int, default=None,
                     help="Linear LR warmup steps for the backbone, starting right when it unfreezes "
                          "in Stage 2. Default: 10%% of the remaining (Stage 2) steps (min 10).")
    ap.add_argument("--min_lr_frac", type=float, default=0.05,
                     help="Cosine decay floor, as a fraction of peak LR")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--warmup_frozen_epochs", type=int, default=3,
                     help="Stage 1: train only the new decoder head with the CLIP backbone fully "
                          "frozen (pure feature extraction) for this many epochs")
    ap.add_argument("--unfreeze_last_n", type=int, default=2,
                     help="Stage 2 (after warmup_frozen_epochs): unfreeze this many of CLIP's final "
                          "transformer blocks and fine-tune them at --backbone_lr")
    ap.add_argument("--augment", dest="augment", action="store_true", default=True,
                     help="Apply training-time augmentation (rotation/perspective/blur/color jitter). "
                          "On by default — helps generalization, especially on smaller datasets.")
    ap.add_argument("--no_augment", dest="augment", action="store_false")
    ap.add_argument("--class_weights", dest="class_weights", action="store_true", default=True,
                     help="Weight the loss by inverse character frequency, to help the model learn "
                          "rare long-tail conjunct consonants instead of mostly just common characters. "
                          "On by default.")
    ap.add_argument("--no_class_weights", dest="class_weights", action="store_false")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                     help="Mixed-precision training on CUDA (faster, lower memory on a single GPU). "
                          "Automatically a no-op on CPU.")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--patience", type=int, default=8,
                     help="Stop early if val CER doesn't improve for this many consecutive epochs (0 disables)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--compile", dest="compile_model", action="store_true", default=False,
                     help="Wrap the model with torch.compile() for extra single-GPU speedup "
                          "(PyTorch 2.0+, CUDA only). Off by default since it adds startup "
                          "compile time and isn't supported on every setup; try it once the "
                          "training loop itself is working.")
    ap.add_argument("--out_dir", default="checkpoints",
                     help="Training different languages? Use a separate directory per language "
                          "(e.g. --out_dir checkpoints/hindi) so checkpoints don't overwrite each other.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yes", "-y", action="store_true",
                     help="Skip the confirmation prompt and start training immediately.")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from <out_dir>/resume_state.pt if it exists (full training state: "
                          "model, optimizer, scheduler, epoch, best CER, history — not just weights). "
                          "Without this flag, training always starts fresh from epoch 1, even if a "
                          "resume_state.pt is present.")
    ap.add_argument("--resume_from", default=None,
                     help="Resume from a specific checkpoint file instead of the default "
                          "<out_dir>/resume_state.pt — e.g. one of the periodic "
                          "checkpoint_epoch_<N>.pt snapshots, to roll back further than the latest "
                          "epoch. Implies --resume.")
    ap.add_argument("--checkpoint_every", type=int, default=5,
                     help="Also save a full resumable snapshot every N epochs to "
                          "<out_dir>/checkpoint_epoch_<N>.pt, which is NEVER overwritten (unlike "
                          "resume_state.pt, which is replaced every epoch) — a rollback point in case "
                          "training needs resuming from further back than the single latest epoch, or "
                          "resume_state.pt itself is ever corrupted. Set to 0 to disable.")
    args = ap.parse_args()

    print_all_params(args)
    confirm_or_exit(args)
    start_time = time.time()

    train_image_root = args.train_image_root or args.image_root
    val_image_root = args.val_image_root or args.image_root
    if not train_image_root:
        raise ValueError("Provide --train_image_root (or --image_root as a shared fallback).")
    if not val_image_root:
        raise ValueError("Provide --val_image_root (or --image_root as a shared fallback).")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"train_image_root: {train_image_root}")
    print(f"val_image_root:   {val_image_root}")
    device = torch.device(args.device)
    log_device(device, "train.py")
    pin_memory = device.type == "cuda"

    if device.type == "cuda":
        # Free speedups on a single CUDA GPU, no accuracy tradeoff worth
        # worrying about for this task: TF32 uses a faster reduced-precision
        # matmul path on Ampere+ GPUs, and cudnn.benchmark lets cuDNN pick
        # faster convolution algorithms once it sees our fixed input shape
        # (all images are letterboxed to the same resolution, so there's no
        # downside to the extra tuning pass on the first batch).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    vocab = build_vocab(args)
    print(f"Vocab size: {len(vocab)}")

    # Stage 1 starts with the backbone fully frozen, regardless of
    # --unfreeze_last_n — unfreezing happens on schedule inside the loop.
    model = CLIPSceneTextOCR(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        clip_checkpoint_path=args.clip_ckpt,
        max_len=args.max_len,
        freeze_backbone=True,
        unfreeze_last_n=0,
        device=device,
    ).to(device)

    preprocess = model.preprocess
    train_ds = SceneTextDataset(args.train_ann, train_image_root, vocab, preprocess,
                                       args.max_len, augment=args.augment)
    val_ds = SceneTextDataset(args.val_ann, val_image_root, vocab, preprocess,
                                     args.max_len, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id), num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0), prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id), num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0), prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    if args.compile_model and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile() enabled.")
        except Exception as e:
            print(f"torch.compile() failed ({e}); continuing without it.")

    # Discriminative learning rates: the pretrained CLIP backbone gets a
    # much smaller LR than the freshly-initialized decoder/head.
    # Params with requires_grad=False simply get skipped by the optimizer
    # step until unfreeze_backbone_last_n() flips them back on.
    backbone_params = list(model.encoder.visual.parameters())
    head_params = (
        list(model.memory_proj.parameters())
        + list(model.token_embedding.parameters())
        + list(model.decoder.parameters())
        + list(model.output_proj.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},   # group 0
            {"params": head_params, "lr": args.lr},                 # group 1
        ],
        weight_decay=1e-4,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    unfreeze_step = steps_per_epoch * args.warmup_frozen_epochs
    stage2_steps = max(1, total_steps - unfreeze_step)

    head_warmup_steps = args.head_warmup_steps
    if head_warmup_steps is None:
        head_warmup_steps = max(20, round(0.1 * total_steps))
    backbone_warmup_steps = args.backbone_warmup_steps
    if backbone_warmup_steps is None:
        backbone_warmup_steps = max(10, round(0.1 * stage2_steps))

    print(f"Steps/epoch: {steps_per_epoch}  Total steps: {total_steps}  "
          f"Head warmup: {head_warmup_steps} steps  Backbone warmup: {backbone_warmup_steps} steps "
          f"(starting at step {unfreeze_step})")
    if head_warmup_steps >= total_steps * 0.5:
        print(f"  !! WARNING: head warmup ({head_warmup_steps} steps) is a large fraction of total "
              f"training ({total_steps} steps) — the LR will barely reach its peak before training "
              f"ends/early-stops. Lower --head_warmup_steps or increase --epochs.")

    # Early stopping must never be able to fire while the LR is still
    # ramping up (or shortly after) — the model hasn't had a fair chance
    # yet, and whatever CER it happened to hit during warmup is not a
    # meaningful "best". Without this floor, a longer --epochs run (which
    # under the adaptive 10%-of-total-steps warmup default also gets a
    # proportionally longer warmup) can trigger --patience almost
    # immediately after warmup completes, stopping training before real
    # learning at full LR ever gets a chance to happen.
    warmup_epochs_equiv = math.ceil((unfreeze_step + backbone_warmup_steps) / steps_per_epoch)
    min_epochs_before_stop = max(args.warmup_frozen_epochs,
                                  math.ceil(head_warmup_steps / steps_per_epoch),
                                  warmup_epochs_equiv) + 5
    if args.patience > 0:
        print(f"Early stopping: disabled until epoch {min_epochs_before_stop} "
              f"(warmup must fully complete first), then --patience={args.patience}.")

    backbone_lambda = make_lr_lambda(backbone_warmup_steps, total_steps,
                                      start_step=unfreeze_step, min_lr_frac=args.min_lr_frac)
    head_lambda = make_lr_lambda(head_warmup_steps, total_steps,
                                  start_step=0, min_lr_frac=args.min_lr_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[backbone_lambda, head_lambda])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    except TypeError:  # older torch without the device-qualified constructor
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    class_weights = compute_class_weights(vocab, args.train_ann).to(device) if args.class_weights else None
    if class_weights is not None:
        print(f"Using class-weighted loss (min={class_weights.min():.3f}, max={class_weights.max():.3f}) "
              f"to help long-tail conjunct consonants.")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.1, weight=class_weights)

    best_cer = float("inf")
    best_epoch = None
    epochs_since_improve = 0
    backbone_unfrozen = False
    epoch_history = []
    stopped_early = False
    start_epoch = 1
    prior_elapsed_seconds = 0.0

    resume_path = args.resume_from or os.path.join(args.out_dir, "resume_state.pt")
    effective_resume = args.resume or bool(args.resume_from)

    if not effective_resume and os.path.isfile(resume_path):
        print(f"Found an existing resume_state.pt at {resume_path} (--resume was not given on the "
              f"command line).")
        if args.yes:
            print("(--yes given) Defaulting to RESUME from this checkpoint rather than starting fresh "
                  "from epoch 1 — pass --resume explicitly next time to skip this message, or delete "
                  "resume_state.pt first if you genuinely want to start fresh.")
            effective_resume = True
        else:
            try:
                answer = input("Resume from this checkpoint instead of starting fresh from epoch 1? "
                                "[Y/n]: ").strip().lower()
            except EOFError:
                print("No interactive input available — defaulting to RESUME (safer than silently "
                      "discarding prior progress). Pass --yes explicitly to suppress this prompt in "
                      "non-interactive/automated runs.")
                answer = "y"
            if answer in ("", "y", "yes"):
                effective_resume = True
            else:
                print("Starting FRESH from epoch 1 — resume_state.pt will be OVERWRITTEN once this run "
                      "saves its first epoch.")
                effective_resume = False

    if effective_resume:
        if not os.path.isfile(resume_path):
            print(f"--resume given but no resume_state.pt found at {resume_path} "
                  f"— starting fresh from epoch 1.")
        else:
            print(f"Resuming from {resume_path} ...")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_cer = ckpt["best_cer"]
            best_epoch = ckpt["best_epoch"]
            epochs_since_improve = ckpt["epochs_since_improve"]
            backbone_unfrozen = ckpt["backbone_unfrozen"]
            epoch_history = ckpt["epoch_history"]
            prior_elapsed_seconds = ckpt.get("elapsed_seconds", 0.0)
            if backbone_unfrozen:
                # Restore which backbone blocks were unfrozen -- requires_grad isn't
                # part of the optimizer/model state_dicts, so this has to be redone
                # explicitly. The optimizer's param groups already cover every
                # parameter regardless of requires_grad (see construction above),
                # so this is safe to call after loading optimizer state.
                model.unfreeze_backbone_last_n(args.unfreeze_last_n)
            print(f"Resumed at epoch {start_epoch}  (best CER so far: {best_cer:.4f} at epoch {best_epoch}, "
                  f"{'backbone unfrozen' if backbone_unfrozen else 'backbone still frozen'}, "
                  f"prior training time: {format_duration(prior_elapsed_seconds)})")
            if start_epoch > args.epochs:
                print(f"!! start_epoch ({start_epoch}) is already past --epochs ({args.epochs}) — "
                      f"nothing to do. Increase --epochs if you want to keep training this run.")

    for epoch in range(start_epoch, args.epochs + 1):
        if not backbone_unfrozen and epoch > args.warmup_frozen_epochs:
            model.unfreeze_backbone_last_n(args.unfreeze_last_n)
            backbone_unfrozen = True
            print(f"[Epoch {epoch}] Stage 2: unfroze last {args.unfreeze_last_n} CLIP block(s), "
                  f"backbone LR warming up to {args.backbone_lr} over {backbone_warmup_steps} steps")

        train_loss = run_epoch(model, train_loader, optimizer, criterion, device,
                                scaler, scheduler, args.grad_accum_steps, args.amp, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device,
                              scaler, None, args.grad_accum_steps, args.amp, train=False)
        cer = evaluate_cer(model, val_loader, vocab, device, args.max_len, use_amp=args.amp)

        cur_lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CER={cer:.4f} "
              f"lr(backbone/head)={cur_lrs[0]:.2e}/{cur_lrs[1]:.2e}")

        epoch_history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_cer": cer,
            "lr_backbone": cur_lrs[0], "lr_head": cur_lrs[1],
        })

        torch.save({"model": raw_state_dict(model), "vocab_chars": vocab.chars},
                   os.path.join(args.out_dir, "last.pt"))

        torch.save({
            "model": raw_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_cer": best_cer,
            "best_epoch": best_epoch,
            "epochs_since_improve": epochs_since_improve,
            "backbone_unfrozen": backbone_unfrozen,
            "epoch_history": epoch_history,
            "elapsed_seconds": prior_elapsed_seconds + (time.time() - start_time),
            "vocab_chars": vocab.chars,
        }, resume_path)

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            # A periodic snapshot that is NEVER overwritten (unlike resume_state.pt,
            # replaced every epoch) -- a rollback point further back than "last epoch"
            # if training needs restarting from an earlier, known-good point, or if
            # resume_state.pt itself is ever corrupted (e.g. a crash mid-write).
            periodic_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model": raw_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_cer": best_cer,
                "best_epoch": best_epoch,
                "epochs_since_improve": epochs_since_improve,
                "backbone_unfrozen": backbone_unfrozen,
                "epoch_history": epoch_history,
                "elapsed_seconds": prior_elapsed_seconds + (time.time() - start_time),
                "vocab_chars": vocab.chars,
            }, periodic_path)
            print(f"  -> periodic snapshot saved: {periodic_path}")

        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({"model": raw_state_dict(model), "vocab_chars": vocab.chars},
                       os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best CER {best_cer:.4f}, checkpoint saved.")
        else:
            epochs_since_improve += 1

        if epoch == min_epochs_before_stop:
            # Give early stopping a clean, full --patience-epoch window
            # starting here, regardless of what happened during warmup —
            # otherwise an already-exhausted counter could trigger a stop
            # the moment this threshold is crossed.
            epochs_since_improve = 0

        if args.patience > 0 and epoch > min_epochs_before_stop and epochs_since_improve >= args.patience:
            print(f"[Epoch {epoch}] val CER hasn't improved for {epochs_since_improve} epochs "
                  f"— stopping early (best={best_cer:.4f}).")
            stopped_early = True
            break

    elapsed_seconds = prior_elapsed_seconds + (time.time() - start_time)
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f" Training samples   : {len(train_ds)}")
    print(f" Validation samples : {len(val_ds)}")
    print(f" Wall-clock time    : {format_duration(elapsed_seconds)}")
    print(f" Epochs run         : {len(epoch_history)} (configured max: {args.epochs}, "
          f"stopped early: {stopped_early})")
    if epoch_history:
        print(f" Best val_CER       : {best_cer:.4f}  (epoch {best_epoch})")
        print(f" train_loss         : {epoch_history[0]['train_loss']:.4f} (epoch 1) -> "
              f"{epoch_history[-1]['train_loss']:.4f} (epoch {epoch_history[-1]['epoch']})")
    print("=" * 60)

    summary_path = os.path.join(args.out_dir, "training_summary.txt")
    json_path = write_summary_file(summary_path, args, len(train_ds), len(val_ds), elapsed_seconds,
                                    epoch_history, best_cer, best_epoch, stopped_early)
    print(f"\nFull summary written to:\n  {summary_path}\n  {json_path}")


if __name__ == "__main__":
    main()
