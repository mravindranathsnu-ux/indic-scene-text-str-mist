"""
inference.py
Run a trained CLIPSceneTextOCR model on new images.

Examples:
    python inference.py --ckpt checkpoints/best.pt --clip_ckpt ./weights/ViT-B-32.pt --image sample.jpg
    python inference.py --ckpt checkpoints/best.pt --clip_ckpt ./weights/ViT-B-32.pt --image_dir samples/ --decode beam --beam_size 5 --out_csv results/predictions.csv
"""
import argparse
import csv
import os
import torch
from PIL import Image

from vocab import Vocab
from model import CLIPSceneTextOCR
from utils import log_device


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
def predict_image(model, vocab, image_path, device, max_len, decode="greedy", beam_size=5,
                   use_amp=True, length_penalty=0.7):
    return predict_batch(model, vocab, [image_path], device, max_len, decode, beam_size,
                          use_amp, length_penalty)[0]


@torch.no_grad()
def predict_batch(model, vocab, image_paths, device, max_len, decode="greedy", beam_size=5,
                   use_amp=True, length_penalty=0.7):
    """Batches multiple images through the model in one forward pass —
    much better single-GPU utilization than looping image-by-image,
    since a single image leaves most of the GPU idle."""
    tensors = [model.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(tensors).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        if decode == "beam":
            texts = model.beam_search_decode(batch, vocab, max_len=max_len, beam_size=beam_size,
                                               length_penalty=length_penalty, device=device)
        else:
            texts = model.greedy_decode(batch, vocab, max_len=max_len, device=device)
    return texts


def print_all_params(args):
    groups = {
        "Checkpoint / model": ["ckpt", "clip_ckpt", "max_len"],
        "Input": ["image", "image_dir"],
        "Decoding": ["decode", "beam_size", "length_penalty"],
        "Run settings": ["batch_size", "amp", "device"],
        "Output": ["out_csv"],
    }
    args_dict = vars(args)
    seen = set()
    print("\n" + "=" * 60)
    print(" INFERENCE PARAMETERS")
    print("=" * 60)
    for group_name, keys in groups.items():
        present = [k for k in keys if k in args_dict]
        if not present:
            continue
        print(f"\n[{group_name}]")
        for k in present:
            print(f"  --{k:<14s} {args_dict[k]}")
            seen.add(k)
    leftover = [k for k in args_dict if k not in seen]
    if leftover:
        print(f"\n[Other]")
        for k in sorted(leftover):
            print(f"  --{k:<14s} {args_dict[k]}")
    print("=" * 60 + "\n")


def check_files_before_running(ckpt, clip_ckpt, image, image_dir):
    print("=" * 60)
    print(" FILE CHECKS")
    print("=" * 60)
    all_ok = True

    for label, path in [("--ckpt", ckpt), ("--clip_ckpt", clip_ckpt)]:
        ok = os.path.isfile(path)
        print(f"  [{'OK' if ok else 'MISSING'}] {label:<12s} {path}")
        all_ok = all_ok and ok

    if image:
        ok = os.path.isfile(image)
        print(f"  [{'OK' if ok else 'MISSING'}] {'--image':<12s} {image}")
        all_ok = all_ok and ok

    if image_dir:
        ok = os.path.isdir(image_dir)
        print(f"  [{'OK' if ok else 'MISSING'}] {'--image_dir':<12s} {image_dir}")
        all_ok = all_ok and ok

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
        answer = input("Proceed with inference using the above? [y/N]: ").strip().lower()
    except EOFError:
        print("\nNo interactive input available — re-run with --yes to skip confirmation "
              "in non-interactive contexts.")
        raise SystemExit(1)
    if answer not in ("y", "yes"):
        print("Aborted — no inference run.")
        raise SystemExit(0)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip_ckpt", required=True,
                     help="Path to local CLIP checkpoint file, e.g. ./weights/ViT-B-32.pt or ViT-B-16.pt")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy",
                     help="'beam' usually gives a small accuracy bump over 'greedy' at the cost of "
                          "slower inference — worth trying once training itself is solid.")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.7,
                     help="Beam search length normalization exponent (only used with --decode beam). "
                          "Higher (e.g. 1.0-1.5) penalizes short/early-EOS completions more.")
    ap.add_argument("--batch_size", type=int, default=16,
                     help="How many images to run through the model at once when using --image_dir "
                          "— batching keeps a single GPU busy instead of processing one image at a time.")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                     help="Mixed-precision inference on CUDA (faster). No-op on CPU.")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--image", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--out_csv", default=None,
                     help="Optional path to save all predictions (image_path,predicted_text) as a CSV. "
                          "Terminal output still prints either way.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yes", "-y", action="store_true",
                     help="Skip the confirmation prompt and start inference immediately.")
    args = ap.parse_args()

    if not args.image and not args.image_dir:
        print("Nothing to do — pass --image <file> or --image_dir <folder>.")
        return

    check_files_before_running(args.ckpt, args.clip_ckpt, args.image, args.image_dir)
    print_all_params(args)
    confirm_or_exit(args)

    device = torch.device(args.device)
    log_device(device, "inference.py")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model, vocab = load_model(args.ckpt, args.clip_ckpt, args.max_len, device)

    results = []  # (path, predicted_text) pairs, written to --out_csv if given

    if args.image:
        text = predict_image(model, vocab, args.image, device, args.max_len, args.decode,
                              args.beam_size, args.amp, args.length_penalty)
        print(f"{args.image}\t{text}")
        results.append((args.image, text))

    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"!! --image_dir '{args.image_dir}' does not exist or is not a directory. No output produced.")
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        fnames = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(exts))

        if not fnames:
            all_files = os.listdir(args.image_dir)
            print(f"!! No images with a supported extension {exts} found in '{args.image_dir}'.")
            if all_files:
                print(f"   Directory contains {len(all_files)} file(s), e.g.: {sorted(all_files)[:5]}")
                print(f"   If those ARE images, their extension isn't in the supported list above.")
            else:
                print(f"   Directory is empty.")
            return

        print(f"Found {len(fnames)} image(s) in '{args.image_dir}'.")
        for i in range(0, len(fnames), args.batch_size):
            chunk = fnames[i:i + args.batch_size]
            paths = [os.path.join(args.image_dir, f) for f in chunk]
            texts = predict_batch(model, vocab, paths, device, args.max_len, args.decode,
                                   args.beam_size, args.amp, args.length_penalty)
            for fname, text in zip(chunk, texts):
                print(f"{fname}\t{text}")
                results.append((fname, text))

    if args.out_csv and results:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "predicted_text"])
            writer.writerows(results)
        print(f"\nSaved {len(results)} prediction(s) to: {args.out_csv}")


if __name__ == "__main__":
    main()
