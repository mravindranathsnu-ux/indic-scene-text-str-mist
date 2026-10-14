"""
evaluate.py
Runs a trained CLIPSceneTextOCR checkpoint over a labeled test set and reports
overall accuracy: Word Recognition Rate (WRR) and Character Recognition
Rate (CRR), plus the underlying CER/WER for reference. Optionally dumps
every per-sample prediction to a CSV for manual error analysis.

Usage:
    python evaluate.py \
        --ckpt checkpoints/best.pt \
        --clip_ckpt /path/to/ViT-B-32.pt \
        --test_ann data/test.tsv \
        --image_root data/images \
        --out_csv results/test_predictions.csv
"""
import argparse
import csv
import json
import os
import time
import torch
from torch.utils.data import DataLoader

from vocab import Vocab
from dataset import SceneTextDataset, collate_fn
from model import CLIPSceneTextOCR
from utils import compute_all_metrics, log_device


def load_model(ckpt_path, clip_ckpt, max_len, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab(chars=ckpt["vocab_chars"])

    model = CLIPSceneTextOCR(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        clip_checkpoint_path=clip_ckpt,
        max_len=max_len,
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab


@torch.no_grad()
def run_inference(model, loader, vocab, device, max_len, decode="greedy", beam_size=5,
                   use_amp=True, length_penalty=0.7):
    all_preds, all_refs = [], []
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            if decode == "beam":
                preds = model.beam_search_decode(images, vocab, max_len=max_len, beam_size=beam_size,
                                                   length_penalty=length_penalty, device=device)
            else:
                preds = model.greedy_decode(images, vocab, max_len=max_len, device=device)
            all_preds.extend(preds)
            all_refs.extend(batch["labels"])
    return all_preds, all_refs


def print_report(metrics):
    print("=" * 46)
    print(" SCENE TEXT RECOGNITION — EVALUATION REPORT")
    print("=" * 46)
    print(f" Samples evaluated               : {metrics['num_samples']}")
    print(f" Word Recognition Rate  (WRR)    : {metrics['WRR']*100:6.2f} %")
    print(f" Character Recognition Rate (CRR): {metrics['CRR']*100:6.2f} %")
    print("-" * 46)
    print(f" Word Error Rate       (WER)     : {metrics['WER']*100:6.2f} %")
    print(f" Character Error Rate  (CER)     : {metrics['CER']*100:6.2f} %")
    print("=" * 46)


def print_all_params(args, test_image_root):
    groups = {
        "Checkpoint / model": ["ckpt", "clip_ckpt", "max_len"],
        "Data": ["test_ann", "test_image_root"],
        "Decoding": ["decode", "beam_size", "length_penalty"],
        "Run settings": ["batch_size", "num_workers", "amp", "device"],
        "Output": ["out_csv", "out_report"],
    }
    args_dict = dict(vars(args))
    args_dict["test_image_root"] = test_image_root  # show the RESOLVED value, not the raw alias args
    seen = set()
    print("\n" + "=" * 60)
    print(" EVALUATION PARAMETERS")
    print("=" * 60)
    for group_name, keys in groups.items():
        present = [k for k in keys if k in args_dict]
        if not present:
            continue
        print(f"\n[{group_name}]")
        for k in present:
            print(f"  --{k:<18s} {args_dict[k]}")
            seen.add(k)
    leftover = [k for k in args_dict if k not in seen and k not in ("image_root",)]
    if leftover:
        print(f"\n[Other]")
        for k in sorted(leftover):
            print(f"  --{k:<18s} {args_dict[k]}")
    print("=" * 60 + "\n")


def check_files_before_running(ckpt, clip_ckpt, test_ann, test_image_root):
    """Checks every file/dir this run depends on BEFORE any GPU/model work
    happens, and spot-checks that test_ann's image paths actually resolve
    under test_image_root — the exact class of bug (wrong language's files,
    duplicated path segments) that has bitten this project repeatedly."""
    print("=" * 60)
    print(" FILE CHECKS")
    print("=" * 60)
    all_ok = True

    for label, path in [("--ckpt", ckpt), ("--clip_ckpt", clip_ckpt), ("--test_ann", test_ann)]:
        ok = os.path.isfile(path)
        print(f"  [{'OK' if ok else 'MISSING'}] {label:<14s} {path}")
        all_ok = all_ok and ok

    ok = os.path.isdir(test_image_root)
    print(f"  [{'OK' if ok else 'MISSING'}] {'--test_image_root':<14s} {test_image_root}")
    all_ok = all_ok and ok

    if os.path.isfile(test_ann) and os.path.isdir(test_image_root):
        sample_lines = []
        with open(test_ann, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    sample_lines.append(parts[0])
                if len(sample_lines) >= 5:
                    break
        found = sum(1 for p in sample_lines if os.path.isfile(os.path.join(test_image_root, p)))
        print(f"\n  Spot-check: {found}/{len(sample_lines)} sample image(s) from --test_ann "
              f"resolve under --test_image_root.")
        if sample_lines and found == 0:
            print(f"  !! NONE of the sampled image paths resolved. First sample path: "
                  f"{sample_lines[0]!r} joined with test_image_root gives: "
                  f"{os.path.join(test_image_root, sample_lines[0])!r}")
            print(f"  This usually means --test_ann/--test_image_root are mismatched (e.g. wrong "
                  f"language, or a duplicated 'images/' path segment) — check before proceeding.")
            all_ok = False
        elif sample_lines and found < len(sample_lines):
            print(f"  !! Only some sample paths resolved — worth double-checking before proceeding.")

    print("=" * 60 + "\n")
    if not all_ok:
        print("!! One or more required files/paths were not found (see MISSING above). "
              "Fix these before continuing — the run will fail otherwise.")
        raise SystemExit(1)


def confirm_or_exit(args):
    if args.yes:
        print("(--yes given, skipping confirmation prompt)\n")
        return
    try:
        answer = input("Proceed with evaluation using the above? [y/N]: ").strip().lower()
    except EOFError:
        print("\nNo interactive input available — re-run with --yes to skip confirmation "
              "in non-interactive contexts.")
        raise SystemExit(1)
    if answer not in ("y", "yes"):
        print("Aborted — no evaluation run.")
        raise SystemExit(0)
    print()


def write_eval_report(path, args, metrics, test_image_root, elapsed_seconds):
    lines = []
    lines.append("=" * 46)
    lines.append(" SCENE TEXT RECOGNITION — EVALUATION REPORT")
    lines.append("=" * 46)
    lines.append(f" Generated               : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f" Checkpoint              : {args.ckpt}")
    lines.append(f" Test set                : {args.test_ann}")
    lines.append(f" Decode                  : {args.decode}"
                  + (f" (beam_size={args.beam_size}, length_penalty={args.length_penalty})"
                     if args.decode == "beam" else ""))
    lines.append(f" Evaluation time         : {elapsed_seconds:.1f}s")
    lines.append(" Samples evaluated               : " + str(metrics["num_samples"]))
    lines.append(f" Word Recognition Rate  (WRR)    : {metrics['WRR']*100:6.2f} %")
    lines.append(f" Character Recognition Rate (CRR): {metrics['CRR']*100:6.2f} %")
    lines.append("-" * 46)
    lines.append(f" Word Error Rate       (WER)     : {metrics['WER']*100:6.2f} %")
    lines.append(f" Character Error Rate  (CER)     : {metrics['CER']*100:6.2f} %")
    lines.append("=" * 46)

    lines.append("")
    lines.append("-" * 46)
    lines.append(" ALL PARAMETERS")
    lines.append("-" * 46)
    for k, v in sorted(vars(args).items()):
        lines.append(f"  --{k:<18s} {v}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": {k: v for k, v in vars(args).items()},
            "test_image_root": test_image_root,
            "elapsed_seconds": elapsed_seconds,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    return json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (e.g. checkpoints/best.pt)")
    ap.add_argument("--clip_ckpt", required=True, help="Path to local CLIP visual backbone .pt file")
    ap.add_argument("--test_ann", required=True, help="TSV file: image_path<TAB>label")
    ap.add_argument("--image_root", default=None, help="Alias for --test_image_root")
    ap.add_argument("--test_image_root", default=None,
                     help="Folder --test_ann's image paths are relative to. Defaults to --image_root.")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy",
                     help="'beam' usually gives a small WRR/CRR bump over 'greedy', at the cost of "
                          "slower evaluation.")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.7,
                     help="Beam search length normalization exponent (only used with --decode beam). "
                          "Higher values (e.g. 1.0-1.5) penalize short/early-EOS completions more — "
                          "try raising this if predictions are systematically truncated/premature-EOS "
                          "on longer words (check with a length-bucketed WRR/CRR breakdown). 0 = no "
                          "length normalization at all.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                     help="Mixed-precision evaluation on CUDA (faster). No-op on CPU.")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--out_csv", default=None, help="Optional path to save every prediction for error analysis")
    ap.add_argument("--out_report", default=None,
                     help="Optional path to save the summary report (metrics + parameters) as a text "
                          "file, plus a machine-readable .json alongside it. Defaults to alongside "
                          "--out_csv (e.g. results/x_predictions.csv -> results/x_report.txt) if "
                          "--out_csv is given but this isn't.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yes", "-y", action="store_true",
                     help="Skip the confirmation prompt and start evaluation immediately.")
    args = ap.parse_args()

    test_image_root = args.test_image_root or args.image_root
    if not test_image_root:
        raise ValueError("Provide --test_image_root (or --image_root as an alias).")

    check_files_before_running(args.ckpt, args.clip_ckpt, args.test_ann, test_image_root)
    print_all_params(args, test_image_root)
    confirm_or_exit(args)

    device = torch.device(args.device)
    log_device(device, "evaluate.py")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model, vocab = load_model(args.ckpt, args.clip_ckpt, args.max_len, device)

    test_ds = SceneTextDataset(args.test_ann, test_image_root, vocab, model.preprocess, args.max_len)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, vocab.pad_id), num_workers=args.num_workers,
    )
    # shuffle=False keeps loader order aligned with test_ds.samples, needed
    # to attach image paths back onto predictions for the CSV dump below.
    image_paths = [img_path for img_path, _ in test_ds.samples]

    start_time = time.time()
    preds, refs = run_inference(model, test_loader, vocab, device, args.max_len,
                                 args.decode, args.beam_size, args.amp, args.length_penalty)
    elapsed_seconds = time.time() - start_time
    metrics = compute_all_metrics(preds, refs)
    print_report(metrics)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "ground_truth", "prediction", "exact_match"])
            for path, ref, pred in zip(image_paths, refs, preds):
                writer.writerow([path, ref, pred, ref == pred])
        print(f"\nPer-sample predictions saved to: {args.out_csv}")

    out_report = args.out_report
    if out_report is None and args.out_csv:
        base, _ext = os.path.splitext(args.out_csv)
        out_report = base.replace("_predictions", "") + "_report.txt"

    if out_report:
        os.makedirs(os.path.dirname(out_report) or ".", exist_ok=True)
        json_path = write_eval_report(out_report, args, metrics, test_image_root, elapsed_seconds)
        print(f"Full report saved to: {out_report}\n                       {json_path}")


if __name__ == "__main__":
    main()
