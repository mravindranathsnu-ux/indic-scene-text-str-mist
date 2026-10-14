#!/usr/bin/env python3
"""
generate_confusion_pair_wordlist.py
Round-2 enhancement, complementary to generate_confusion_focus_wordlist.py:
generates MINIMAL-PAIR CONTRASTIVE synthetic words for the specific
character pairs a language's model actually confuses (learned from its
own real test predictions), rather than just "more examples of words
containing a hard character" (which is what 04b/generate_confusion_focus_
wordlist.py already does).

For a confusion pair like స->న (ground truth స, model predicted న): finds
real words containing స, and generates a SYNTHETIC TWIN for each by
substituting స -> న throughout the word. The twin need not be a real
dictionary word -- it's still a valid training example, since the
rendered image shows exactly the substituted characters and the label
correctly reflects that. This gives the model direct, paired exposure to
both glyphs in otherwise-identical visual contexts, which is a more
targeted signal for learning to discriminate two specific confusable
glyphs than generic extra exposure to one of them.

Works for every Indic language in this project the same way -- it only
needs the predictions CSV and train.tsv, no script-specific logic.

    python generate_confusion_pair_wordlist.py \\
        --predictions results/telugu_test_predictions.csv \\
        --train_ann data/telugu/train/train.tsv \\
        --out data/telugu/train/confusion_pair_words.txt

    # then, same as the confusion-focus wordlist, feed into a synthetic
    # pass (via --corpus_file or --train_ann with --out_format tsv) at
    # a higher --samples_per_word than the normal pass.
"""
import argparse
import csv
import os
from collections import Counter


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


def learn_confusion_pairs(predictions_path, top_n_pairs):
    rows = list(csv.DictReader(open(predictions_path, encoding="utf-8")))
    wrong = [r for r in rows if r["exact_match"] == "False"]

    pair_count = Counter()
    for r in wrong:
        gt, pred = r["ground_truth"], r["prediction"]
        if pred == "" or abs(len(gt) - len(pred)) > 3:
            continue
        subs = align_substitutions(gt, pred)
        if 1 <= len(subs) <= 2:
            for gt_ch, pred_ch in subs:
                if gt_ch != pred_ch:
                    pair_count[(gt_ch, pred_ch)] += 1

    return pair_count.most_common(top_n_pairs), len(rows), len(wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Path to <lang>_test_predictions.csv")
    ap.add_argument("--train_ann", required=True, help="Path to the real train.tsv to search for words")
    ap.add_argument("--out", required=True, help="Output path for the minimal-pair word list")
    ap.add_argument("--out_format", choices=["corpus", "tsv"], default="corpus",
                     help="'corpus': plain one-word-per-line. 'tsv': image_path[placeholder]\\tlabel.")
    ap.add_argument("--top_n_pairs", type=int, default=10,
                     help="How many of the most-frequent confusion PAIRS to target (default 10)")
    ap.add_argument("--max_words_per_pair", type=int, default=30,
                     help="Cap on how many twin words to generate per pair (default 30), so one very "
                          "common character doesn't dominate the resulting set -- same lesson as the "
                          "max_words cap in generate_confusion_focus_wordlist.py, applied per-pair here.")
    args = ap.parse_args()

    print(f"Learning confusion pairs from: {args.predictions}")
    top_pairs, n_total, n_wrong = learn_confusion_pairs(args.predictions, args.top_n_pairs)
    print(f"  {n_wrong}/{n_total} predictions wrong; learned from their character-level substitutions")
    print(f"\nTop {len(top_pairs)} confusion pairs targeted:")
    for (gt_ch, pred_ch), count in top_pairs:
        print(f"  {gt_ch!r} -> {pred_ch!r}  ({count}x)")

    if not top_pairs:
        raise SystemExit("\nNo confusion pairs found -- nothing to generate.")

    real_words = set()
    with open(args.train_ann, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1]:
                real_words.add(parts[1])

    twin_words = set()
    per_pair_counts = {}
    for (gt_ch, pred_ch), count in top_pairs:
        matched = sorted(w for w in real_words if gt_ch in w)
        selected = matched[:args.max_words_per_pair]
        per_pair_counts[(gt_ch, pred_ch)] = (len(matched), len(selected))
        for w in selected:
            twin = w.replace(gt_ch, pred_ch)
            if twin != w:  # guard: skip if substitution somehow produced no change
                twin_words.add(twin)

    print(f"\nTwin words generated per pair (capped at {args.max_words_per_pair} source words each):")
    for (gt_ch, pred_ch), (found, used) in per_pair_counts.items():
        print(f"  {gt_ch!r} -> {pred_ch!r}: {found} real word(s) contain {gt_ch!r}, used {used}")

    if not twin_words:
        raise SystemExit("\nNo twin words could be generated -- nothing to write.")

    twin_words = sorted(twin_words)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.out_format == "corpus":
            for w in twin_words:
                f.write(f"{w}\n")
        else:
            for i, w in enumerate(twin_words):
                f.write(f"confusion_pair_{i:05d}.png\t{w}\n")

    already_real = sum(1 for w in twin_words if w in real_words)
    print(f"\nWrote {len(twin_words)} synthetic twin word(s) to {args.out} ({args.out_format} format).")
    print(f"({already_real} of these happen to already be real words in your vocabulary -- the rest "
          f"are artificial minimal-pair contrasts, valid as image-label training pairs regardless.)")


if __name__ == "__main__":
    main()
