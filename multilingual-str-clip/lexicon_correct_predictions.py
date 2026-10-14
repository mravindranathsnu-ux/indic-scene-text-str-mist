#!/usr/bin/env python3
"""
lexicon_correct_predictions.py
Post-hoc lexicon correction for scene text predictions -- estimates the
potential impact of lexicon-constrained decoding WITHOUT needing to modify
model.py's beam search loop. Takes an existing evaluate.py --out_csv
(image_path, ground_truth, prediction, exact_match) and a lexicon (built
from TRAIN + VAL vocabulary only, never test -- see --lexicon_ann), and for
every prediction not already a valid lexicon word, looks for the closest
lexicon word by edit distance, substituting it only if the match is close
enough AND unambiguous (a single closest candidate, not a tie).

DESIGNED FOR ALL INDIC LANGUAGES, NOT JUST TELUGU:
Testing on real Telugu data found that lexicon correction can break an
already-correct prediction when the model correctly recognized a valid,
less-common inflected word form that isn't itself in the lexicon, and the
correction "fixes" it into a different inflected variant instead
(e.g. 'వసంత' -> 'వసంతం', 'ఉపాధ్యా' -> 'ఉపాధ్యాయ' -- both real cases from
Telugu's actual test set). This matches a concern raised directly in prior
Indic STR literature (Mathew, Jain & Jawahar 2021): Indic languages are
"highly inflectional," so a fixed lexicon built from training vocabulary
alone may not cover every valid word form.

Rather than building a per-script table of "which characters mark a
grammatical suffix" (fragile -- this project has already hit a real bug
from a wrong per-script Unicode range once, for Gujarati vowel signs, and
would need ~10 separately-verified tables, one per script family, without
native-speaker review), this uses a SCRIPT-AGNOSTIC signal that
generalizes across every Indic script in this project: EDIT POSITION.
Suffixation happens at word edges in both the Dravidian (Telugu, Tamil,
Kannada, Malayalam) and Indo-Aryan (Hindi, Marathi, Gujarati, Bengali,
Odia, Punjabi) families -- a shared structural property, not a
Telugu-specific quirk. An edit confined to the last --suffix_guard_chars
characters is plausibly a genuine morphological variant and is left
uncorrected by default; an edit earlier in the word is far more likely to
be visual glyph confusion (the actual target of this technique) and is
corrected as before. This requires computing the full edit ALIGNMENT (not
just the distance), so edit position can be checked.

Usage:
    python lexicon_correct_predictions.py \\
        --predictions results/telugu_test_predictions.csv \\
        --lexicon_ann data/telugu/train/train.tsv data/telugu/val/val.tsv \\
        --max_edit_distance 1 \\
        --suffix_guard_chars 1
"""
import argparse
import csv
from collections import defaultdict


def align(a, b):
    """Full edit-distance alignment (not just the distance): returns
    (distance, list_of_edit_positions_in_b). Position is measured from the
    END of the longer string (0 = last character), so it's meaningful
    regardless of insertions/deletions shifting absolute indices -- what
    matters here is "how close to the word's edge did this edit happen,"
    not its raw index from the start."""
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

    edit_positions_from_end = []  # distance from the end of b, in characters
    i, j = m, n
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            edit_positions_from_end.append(n - j)
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            edit_positions_from_end.append(n - j)
            j -= 1
    return dp[m][n], edit_positions_from_end


def build_lexicon(ann_paths):
    lexicon = set()
    for path in ann_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[1]:
                    lexicon.add(parts[1])
    return lexicon


def correct(prediction, lexicon, by_length, max_edit_distance, suffix_guard_chars):
    """Returns (corrected_text, was_corrected, reason)."""
    if prediction in lexicon:
        return prediction, False, "already_valid"
    if not prediction:
        return prediction, False, "empty_prediction"

    candidates = []
    for length in range(max(1, len(prediction) - max_edit_distance),
                         len(prediction) + max_edit_distance + 1):
        candidates.extend(by_length.get(length, []))

    scored = []  # (distance, edit_positions, word)
    for word in candidates:
        d, positions = align(prediction, word)
        if d <= max_edit_distance:
            scored.append((d, positions, word))

    if not scored:
        return prediction, False, "no_close_match"

    best_dist = min(s[0] for s in scored)
    best = [s for s in scored if s[0] == best_dist]

    if len(best) > 1:
        return prediction, False, f"ambiguous_{len(best)}_tied_at_dist_{best_dist}"

    _, positions, word = best[0]
    # Script-agnostic suffix guard: if EVERY edit for this candidate falls
    # within the last suffix_guard_chars characters, treat it as plausible
    # morphological variation and decline to correct -- regardless of
    # which script or which specific characters are involved.
    if suffix_guard_chars > 0 and positions and all(p < suffix_guard_chars for p in positions):
        return prediction, False, f"declined_suffix_guard_dist_{best_dist}"

    return word, True, f"corrected_dist_{best_dist}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="evaluate.py --out_csv output")
    ap.add_argument("--lexicon_ann", nargs="+", required=True,
                     help="TSV file(s) to build the lexicon from -- TRAIN and VAL only, "
                          "NEVER test (using test labels here would leak test information "
                          "into the correction step and invalidate the resulting numbers)")
    ap.add_argument("--max_edit_distance", type=int, default=1,
                     help="Only correct predictions within this many character edits of a "
                          "lexicon word (default 1 -- conservative, matches the dominant "
                          "single-character-substitution error type this project has found "
                          "in every language so far)")
    ap.add_argument("--suffix_guard_chars", type=int, default=1,
                     help="Decline to correct when every edit for the best candidate falls "
                          "within this many characters of the word's end (default 1) -- "
                          "script-agnostic guard against 'correcting' a genuine inflected "
                          "form into a different one. Set to 0 to disable (matches the "
                          "original, unguarded behavior).")
    ap.add_argument("--out_csv", default=None,
                     help="Optional: write the corrected predictions to a new CSV")
    args = ap.parse_args()

    print(f"Building lexicon from: {', '.join(args.lexicon_ann)}")
    lexicon = build_lexicon(args.lexicon_ann)
    print(f"Lexicon size: {len(lexicon)} unique word(s)")

    by_length = defaultdict(list)
    for w in lexicon:
        by_length[len(w)].append(w)

    rows = list(csv.DictReader(open(args.predictions, encoding="utf-8")))
    n = len(rows)

    before_correct = sum(1 for r in rows if r["prediction"] == r["ground_truth"])
    after_correct = 0
    n_corrected = 0
    n_corrected_and_fixed = 0
    n_corrected_but_wrong = 0
    n_already_right_left_alone = 0
    reason_counts = defaultdict(int)
    out_rows = []

    for r in rows:
        gt, pred = r["ground_truth"], r["prediction"]
        new_pred, was_corrected, reason = correct(pred, lexicon, by_length, args.max_edit_distance,
                                                    args.suffix_guard_chars)
        reason_counts[reason] += 1

        if was_corrected:
            n_corrected += 1
            if new_pred == gt and pred != gt:
                n_corrected_and_fixed += 1
            elif new_pred != gt and pred == gt:
                n_corrected_but_wrong += 1  # lexicon correction BROKE a previously-correct prediction

        final_correct = (new_pred == gt)
        after_correct += final_correct
        out_rows.append([r["image_path"], gt, pred, new_pred, was_corrected, reason, final_correct])

    print(f"\n{'='*60}")
    print(f" LEXICON CORRECTION IMPACT (max_edit_distance={args.max_edit_distance}, "
          f"suffix_guard_chars={args.suffix_guard_chars})")
    print(f"{'='*60}")
    print(f"Samples: {n}")
    print(f"WRR before correction: {100*before_correct/n:.2f}%  ({before_correct}/{n})")
    print(f"WRR after correction:  {100*after_correct/n:.2f}%  ({after_correct}/{n})")
    print(f"Net change: {100*(after_correct-before_correct)/n:+.2f} percentage points")
    print(f"\nPredictions actually changed by correction: {n_corrected} ({100*n_corrected/n:.1f}%)")
    print(f"  -> changed WRONG to RIGHT:  {n_corrected_and_fixed}")
    print(f"  -> changed RIGHT to WRONG:  {n_corrected_but_wrong}  (should be 0 -- see note below)")
    print(f"  -> changed WRONG to (still) WRONG: {n_corrected - n_corrected_and_fixed - n_corrected_but_wrong}")
    print(f"\nBreakdown by reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:35s} {count:5d}")

    if n_corrected_but_wrong > 0:
        print(f"\n!! {n_corrected_but_wrong} previously-CORRECT prediction(s) got changed to something else.")
        print(f"!! This should be impossible if the ground truth is always in the lexicon -- if it")
        print(f"!! happens, it means the lexicon is missing some test-adjacent real words, or the")
        print(f"!! max_edit_distance is catching a coincidental near-match. Worth checking those")
        print(f"!! specific cases before trusting this correction step.")

    if args.out_csv:
        import os
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "ground_truth", "original_prediction", "corrected_prediction",
                        "was_corrected", "reason", "final_exact_match"])
            w.writerows(out_rows)
        print(f"\nWrote corrected predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
