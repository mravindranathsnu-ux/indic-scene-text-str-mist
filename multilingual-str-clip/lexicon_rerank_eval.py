"""
lexicon_rerank_eval.py
Real lexicon-constrained decoding, using model.py's beam_search_decode_candidates
(added specifically for this) rather than the earlier post-hoc
lexicon_correct_predictions.py approach.

Key difference from the post-hoc script: for each image, this checks
whether ANY of the model's actual top-beam_size hypotheses is a valid
lexicon word -- using real alternatives the model itself considered
during search, not blind edit-distance guesses against a single final
string. Only if NONE of the beam candidates are valid lexicon words does
it fall back to the same suffix-guarded nearest-neighbor correction
validated in lexicon_correct_predictions.py.

Standalone script, deliberately NOT wired into evaluate.py yet -- run
this first against a real checkpoint and inspect the results before
considering whether to promote this into the actual evaluation pipeline.

Usage:
    python lexicon_rerank_eval.py \
        --ckpt checkpoints/telugu/best.pt \
        --clip_ckpt weights/ViT-B-16.pt \
        --test_ann data/telugu/test/test.tsv \
        --test_image_root data/telugu/test/images \
        --lexicon_ann data/telugu/train/train.tsv data/telugu/val/val.tsv \
        --beam_size 5 --length_penalty 1.3 \
        --max_edit_distance 1 --suffix_guard_chars 1 \
        --out_csv results/telugu_lexicon_rerank.csv
"""
import argparse
import csv
import os
import time
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from vocab import Vocab
from dataset import SceneTextDataset, collate_fn
from model import CLIPSceneTextOCR
from utils import compute_all_metrics, log_device


def load_model(ckpt_path, clip_ckpt, max_len, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab(chars=ckpt["vocab_chars"])
    model = CLIPSceneTextOCR(
        vocab_size=len(vocab), pad_id=vocab.pad_id, clip_checkpoint_path=clip_ckpt,
        max_len=max_len, device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab


def build_lexicon(ann_paths):
    """Returns (lexicon_set, word_freq_counter) -- frequency is how many
    times each word appears across the given TSV files, used to break
    ties between multiple equally-close correction candidates (prefer the
    more common word) instead of always declining on any tie."""
    from collections import Counter
    word_freq = Counter()
    for path in ann_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[1]:
                    word_freq[parts[1]] += 1
    return set(word_freq.keys()), word_freq


def align(a, b):
    """Full edit-distance alignment (distance + edit positions from the
    end of b) -- identical logic to lexicon_correct_predictions.py, kept
    self-contained here rather than imported, so this script has no
    import-order dependency on the other one."""
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
            # Prefer whichever tied candidate is more common in the real
            # training data -- a reasonable prior when otherwise unable to
            # distinguish, rather than declining every tie outright. Only
            # resolves the tie if there's a single clear-cut most-frequent
            # candidate; if frequency ALSO ties, still decline rather than
            # guess arbitrarily.
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
           word_freq=None, tie_break_by_frequency=True):
    """
    candidates: list of (text, score) sorted best-first from the beam.
    Returns (final_text, method) where method explains how the choice was made.
    """
    top1_text = candidates[0][0]

    # Priority 1: is the model's OWN top choice already a valid word?
    if top1_text in lexicon:
        return top1_text, "top1_already_valid"

    # Priority 2: does ANY other real beam candidate the model actually
    # considered match the lexicon? This is the genuinely new capability
    # over the post-hoc script -- real alternatives, not guessed ones.
    for rank, (text, score) in enumerate(candidates[1:], start=1):
        if text in lexicon:
            return text, f"beam_alternative_rank_{rank}"

    # Priority 3: no beam candidate is a real lexicon word -- fall back to
    # the same suffix-guarded nearest-neighbor correction on the top-1
    # prediction only, exactly as validated in lexicon_correct_predictions.py,
    # now with an optional frequency-aware tie-break.
    corrected, was_corrected, reason = nearest_neighbor_correct(
        top1_text, lexicon, by_length, max_edit_distance, suffix_guard_chars,
        word_freq=word_freq, tie_break_by_frequency=tie_break_by_frequency)
    if was_corrected:
        return corrected, f"nearest_neighbor_{reason}"
    return top1_text, f"kept_top1_{reason}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip_ckpt", required=True)
    ap.add_argument("--test_ann", required=True)
    ap.add_argument("--test_image_root", required=True)
    ap.add_argument("--lexicon_ann", nargs="+", required=True,
                     help="TSV file(s) to build the lexicon from -- TRAIN and VAL only, "
                          "NEVER test (would leak test information into the rerank step)")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.7)
    ap.add_argument("--max_edit_distance", type=int, default=1)
    ap.add_argument("--suffix_guard_chars", type=int, default=1)
    ap.add_argument("--tie_break_by_frequency", dest="tie_break_by_frequency", action="store_true", default=True,
                     help="On a tie between multiple equally-close correction candidates, prefer "
                          "whichever is more common in the training data instead of always declining.")
    ap.add_argument("--no_tie_break_by_frequency", dest="tie_break_by_frequency", action="store_false")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yes", "-y", action="store_true")
    args = ap.parse_args()

    for label, path in [("--ckpt", args.ckpt), ("--clip_ckpt", args.clip_ckpt), ("--test_ann", args.test_ann)]:
        if not os.path.isfile(path):
            raise SystemExit(f"!! Missing required file: {label} = {path}")
    if not os.path.isdir(args.test_image_root):
        raise SystemExit(f"!! Missing required dir: --test_image_root = {args.test_image_root}")
    for path in args.lexicon_ann:
        if not os.path.isfile(path):
            raise SystemExit(f"!! Missing lexicon source: {path}")

    if not args.yes:
        print(f"About to run lexicon-reranked beam decoding:")
        print(f"  ckpt={args.ckpt}  beam_size={args.beam_size}  length_penalty={args.length_penalty}")
        print(f"  lexicon from: {', '.join(args.lexicon_ann)}")
        try:
            ans = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            raise SystemExit("No interactive input available -- re-run with --yes.")
        if ans not in ("y", "yes"):
            raise SystemExit("Aborted.")

    device = torch.device(args.device)
    log_device(device, "lexicon_rerank_eval.py")

    print(f"Building lexicon from: {', '.join(args.lexicon_ann)}")
    lexicon, word_freq = build_lexicon(args.lexicon_ann)
    print(f"Lexicon size: {len(lexicon)} unique word(s)")
    by_length = defaultdict(list)
    for w in lexicon:
        by_length[len(w)].append(w)

    model, vocab = load_model(args.ckpt, args.clip_ckpt, args.max_len, device)

    test_ds = SceneTextDataset(args.test_ann, args.test_image_root, vocab, model.preprocess, args.max_len)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, vocab.pad_id), num_workers=args.num_workers,
    )
    image_paths = [p for p, _ in test_ds.samples]

    print(f"Running beam search (beam_size={args.beam_size}, length_penalty={args.length_penalty}) "
          f"and reranking against the lexicon...")
    start = time.time()
    all_refs, all_top1, all_reranked, all_methods = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                batch_candidates = model.beam_search_decode_candidates(
                    images, vocab, max_len=args.max_len, beam_size=args.beam_size,
                    length_penalty=args.length_penalty, device=device)
            for cand_list, ref in zip(batch_candidates, batch["labels"]):
                top1 = cand_list[0][0]
                final, method = rerank(cand_list, lexicon, by_length, args.max_edit_distance, args.suffix_guard_chars,
                                        word_freq=word_freq, tie_break_by_frequency=args.tie_break_by_frequency)
                all_refs.append(ref)
                all_top1.append(top1)
                all_reranked.append(final)
                all_methods.append(method)
    elapsed = time.time() - start

    metrics_before = compute_all_metrics(all_top1, all_refs)
    metrics_after = compute_all_metrics(all_reranked, all_refs)

    n = len(all_refs)
    n_changed = sum(1 for t, r in zip(all_top1, all_reranked) if t != r)
    n_fixed = sum(1 for t, r, g in zip(all_top1, all_reranked, all_refs) if t != r and r == g)
    n_broken = sum(1 for t, r, g in zip(all_top1, all_reranked, all_refs) if t != r and t == g and r != g)
    method_counts = defaultdict(int)
    for m in all_methods:
        method_counts[m] += 1

    print(f"\n{'='*60}\n LEXICON RERANK RESULTS (beam-aware, {elapsed:.1f}s)\n{'='*60}")
    print(f"Samples: {n}")
    print(f"WRR before (top-1 beam):  {metrics_before['WRR']*100:.2f}%")
    print(f"WRR after (reranked):     {metrics_after['WRR']*100:.2f}%")
    print(f"CRR before (top-1 beam):  {metrics_before['CRR']*100:.2f}%")
    print(f"CRR after (reranked):     {metrics_after['CRR']*100:.2f}%")
    print(f"\nPredictions changed: {n_changed} ({100*n_changed/n:.1f}%)")
    print(f"  -> WRONG to RIGHT: {n_fixed}")
    print(f"  -> RIGHT to WRONG: {n_broken}  (should be 0 or near it)")
    print(f"\nMethod breakdown:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:35s} {c:5d}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "ground_truth", "top1_beam", "reranked", "method", "final_exact_match"])
            for path, gt, top1, final, method in zip(image_paths, all_refs, all_top1, all_reranked, all_methods):
                w.writerow([path, gt, top1, final, method, final == gt])
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
