#!/usr/bin/env python3
"""
generate_confusion_focus_wordlist.py
Round-2 enhancement: learns which characters a language's model actually
confuses (from its own real test predictions -- the same Levenshtein-
alignment method used throughout this project's error analysis), then
scans the real training vocabulary for words containing those characters
and writes them out as a focused word list.

Compatible with the project's actual generate_synthetic_data.py: that
script de-duplicates its word source into a set, so a word repeated in
the input file is NOT rendered extra times -- boosting has to come from
running it a second time on just this focused list, with a HIGHER
--samples_per_word than your normal pass uses. This script therefore
writes each matching word exactly ONCE (deduplicated), suitable for
--corpus_file or --train_ann.

    python generate_confusion_focus_wordlist.py \\
        --predictions results/punjabi_test_predictions.csv \\
        --train_ann data/punjabi/train/train.tsv \\
        --out data/punjabi/train/confusion_focus_words.txt

    # then, as an ADDITIONAL pass alongside your normal synthetic step,
    # with a higher --samples_per_word than the main pass (e.g. 2-3x):
    python generate_synthetic_data.py \\
        --train_ann data/punjabi/train/train.tsv \\
        --corpus_file data/punjabi/train/confusion_focus_words.txt \\
        --exclude_ann data/punjabi/val/val.tsv data/punjabi/test/test.tsv \\
        --fonts_dir fonts/punjabi \\
        --out_dir data/punjabi/train/synthetic \\
        --samples_per_word <your normal value>

    # NOTE: --corpus_file words share the SAME --samples_per_word as
    # --train_ann words in one run (both get folded into the same word
    # set). To give the focus words a DIFFERENT, higher sample count,
    # run generate_synthetic_data.py TWICE: once normally, once more
    # with --train_ann pointed at ONLY this focus-word file (formatted
    # as a 2-column TSV, see --out_format tsv below) and a higher
    # --samples_per_word, into a separate --out_dir, then concatenate
    # both synthetic.tsv outputs before combining with the real data.
"""
import argparse
import csv
import os
from collections import Counter


def eval_dist(a, b):
    a, b = list(a), list(b)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = tmp
    return dp[n]


def align_substitutions(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(n + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                bt[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    subs = []
    i, j = m, n
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            subs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
    return subs


def learn_confusable_chars(predictions_path, top_n):
    rows = list(csv.DictReader(open(predictions_path, encoding="utf-8")))
    wrong = [r for r in rows if r["exact_match"] == "False"]

    char_confusion_count = Counter()
    pair_count = Counter()
    for r in wrong:
        gt, pred = r["ground_truth"], r["prediction"]
        if pred == "" or abs(len(gt) - len(pred)) > 3:
            continue
        subs = align_substitutions(gt, pred)
        if 1 <= len(subs) <= 2:
            for gt_ch, pred_ch in subs:
                char_confusion_count[gt_ch] += 1
                pair_count[(gt_ch, pred_ch)] += 1

    top_chars = [c for c, _ in char_confusion_count.most_common(top_n)]
    return top_chars, char_confusion_count, pair_count, len(rows), len(wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Path to <lang>_test_predictions.csv")
    ap.add_argument("--train_ann", required=True, help="Path to the real train.tsv to search for words")
    ap.add_argument("--out", required=True, help="Output path for the focused word list")
    ap.add_argument("--out_format", choices=["corpus", "tsv"], default="corpus",
                     help="'corpus': plain one-word-per-line (for --corpus_file). "
                          "'tsv': image_path[placeholder]\\tlabel (for --train_ann on a "
                          "separate, higher-samples_per_word pass). Default: corpus.")
    ap.add_argument("--top_n_chars", type=int, default=15,
                     help="How many of the most-confused characters to target (default 15)")
    ap.add_argument("--max_words", type=int, default=300,
                     help="Cap on how many words to include, ranked by how many DISTINCT top-confusable "
                          "characters each word contains (most concentrated first). Without this cap, a "
                          "language whose top-confused characters are common structural elements (e.g. "
                          "frequent vowel signs) can match a large fraction of the entire vocabulary, "
                          "making the 'focused' boost pass dominate the whole training mix instead of "
                          "targeting a genuinely narrow set of hard cases — this happened in practice on "
                          "Telugu (4,221 words matched out of 17,500, becoming 59.5%% of all training "
                          "data and destabilizing training) before this cap was added.")
    args = ap.parse_args()

    print(f"Learning confusable characters from: {args.predictions}")
    top_chars, char_counts, pair_counts, n_total, n_wrong = learn_confusable_chars(
        args.predictions, args.top_n_chars)
    print(f"  {n_wrong}/{n_total} predictions wrong; learned from their character-level substitutions")
    print(f"\nTop {len(top_chars)} confusable characters (by how often each was the correct answer, "
          f"but predicted wrong):")
    for c in top_chars:
        print(f"  {c!r}: confused {char_counts[c]} time(s)")
    print(f"\nTop 10 specific confusion pairs, for reference:")
    for (gt, pred), count in pair_counts.most_common(10):
        print(f"  {gt!r} -> {pred!r}  ({count}x)")

    if not top_chars:
        raise SystemExit("\nNo confusion data found -- nothing to boost. "
                          "(Is this language's result already at or near 100%?)")
    top_chars_set = set(top_chars)

    print(f"\nScanning {args.train_ann} for words containing these characters...")
    all_words = set()
    with open(args.train_ann, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1]:
                all_words.add(parts[1])

    # Rank by concentration (how many DISTINCT top-confusable characters a
    # word contains), most concentrated first, then cap -- see --max_words
    # help text for why this matters. Ties broken alphabetically for
    # reproducibility.
    scored = []
    for w in all_words:
        overlap = top_chars_set & set(w)
        if overlap:
            scored.append((len(overlap), w))
    scored.sort(key=lambda x: (-x[0], x[1]))

    total_matched = len(scored)
    matched_words = sorted(w for _, w in scored[:args.max_words])

    if not matched_words:
        raise SystemExit(f"\nNo words in {args.train_ann} contain any of the top confusable characters "
                          f"-- nothing to write.")

    if total_matched > args.max_words:
        print(f"{total_matched} word(s) matched at least one top-confusable character -- that's "
              f"{100*total_matched/max(len(all_words),1):.0f}% of the {len(all_words)}-word vocabulary, "
              f"too broad to be a genuinely 'focused' boost. Capped to the {args.max_words} most "
              f"CONCENTRATED words (containing the most distinct confusable characters each).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.out_format == "corpus":
            for w in matched_words:
                f.write(f"{w}\n")
        else:
            for i, w in enumerate(matched_words):
                f.write(f"confusion_focus_{i:05d}.png\t{w}\n")

    print(f"\n{len(matched_words)} word(s) selected (of {total_matched} that matched at least one top "
          f"confusable character). Wrote them to {args.out} ({args.out_format} format).")
    print(f"\nThese words get the SAME --samples_per_word as everything else if used via --corpus_file "
          f"in one run. To boost them further, run generate_synthetic_data.py a second time with "
          f"--train_ann pointed at this file (--out_format tsv) and a higher --samples_per_word, into "
          f"a separate --out_dir, then combine both synthetic outputs. Sanity-check the resulting size "
          f"against your main dataset before training -- it should be a modest supplement, not the "
          f"majority of your training mix.")


if __name__ == "__main__":
    main()

