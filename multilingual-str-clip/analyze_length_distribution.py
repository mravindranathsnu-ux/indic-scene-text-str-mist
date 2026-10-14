"""
analyze_length_distribution.py
Analyzes word-length distribution in a training annotation TSV and
suggests a --length_boost value for generate_synthetic_data.py.

This is a HEURISTIC starting point, based on three real data points from
this project so far:
  - Telugu: --length_boost 0.8 was sufficient — no length-related WRR
    decline observed in the eval breakdown.
  - Tamil: --length_boost 0.8 was NOT sufficient — a real WRR decline for
    longer words showed up (75.0% at 1-4 chars down to 44.4% at 13+ chars),
    and Tamil's real training data has a lower fraction of long words than
    Telugu's, which is the most likely explanation.
  - Gujarati: the OPPOSITE failure — its distribution is intrinsically
    short-word-dominated (62% of the test set is 1-4 chars), and the old
    one-directional formula gave it the MAXIMUM long-word boost, which is
    the leading suspect for its uniquely inverted length-accuracy pattern
    (short words worst, the only language showing that). The formula is now
    non-monotonic: boost scales UP for languages with a moderate long-word
    deficit, but scales DOWN below a threshold where the language simply
    doesn't have long words — synthetic data should match such a
    distribution, not fight it.

Always check the length-bucketed WRR/CRR breakdown on actual eval results
after training (see the analysis approach used throughout this project)
and adjust further if the suggested value doesn't fully close the gap.

Usage:
    python analyze_length_distribution.py --train_ann data/tamil/train/train.tsv
    python analyze_length_distribution.py --train_ann data/tamil/train/train.tsv --print_boost_only
"""
import argparse

LONG_THRESHOLD = 9       # matches this project's length-bucket boundaries used in eval analysis
TARGET_LONG_FRAC = 0.30  # heuristic target: fraction of words >= LONG_THRESHOLD chars considered "well-represented"
BASE_BOOST = 0.8         # the value that worked for Telugu
MAX_BOOST = 2.0          # cap to avoid an extreme, untested value
PEAK_FRAC = 0.08         # long-word fraction at which up-boosting peaks; below this, the language is
                         # treated as intrinsically short-word-dominated and the boost scales DOWN
MIN_BOOST = 0.6          # floor for languages with essentially no long words at all


def analyze(train_ann):
    lengths = []
    with open(train_ann, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1]:
                lengths.append(len(parts[1]))

    if not lengths:
        raise ValueError(f"No labels parsed from {train_ann}")

    n = len(lengths)
    avg_len = sum(lengths) / n
    max_len = max(lengths)
    long_count = sum(1 for l in lengths if l >= LONG_THRESHOLD)
    long_frac = long_count / n

    # NON-MONOTONIC response (revised after the Gujarati finding):
    #
    #   - long_frac >= PEAK_FRAC: the language genuinely has long words, and
    #     proportionally fewer of them than the target means the model will
    #     be underserved on a real part of the test distribution -- scale the
    #     boost UP, exactly as before (Telugu ~0.96, Tamil -> 2.0 cap:
    #     unchanged from the previous formula for these validated cases).
    #
    #   - long_frac < PEAK_FRAC: the language is intrinsically short-word-
    #     dominated -- its TEST distribution is short too (same source), so
    #     aggressively boosting long synthetic words starves the short words
    #     that dominate scoring. Observed directly with Gujarati (62% of its
    #     test set is 1-4 chars; its length-bucket accuracy came out INVERTED,
    #     short words worst -- the only language showing that pattern, after
    #     training on maximally long-boosted synthetic data under the old
    #     formula). Scale DOWN linearly toward MIN_BOOST as long_frac
    #     approaches zero: match the real distribution instead of fighting it.
    #
    # Continuous at the boundary: both branches give the same value at
    # PEAK_FRAC. Still a heuristic -- now with three data points behind it
    # (Telugu, Tamil, Gujarati) instead of two.
    if long_frac >= PEAK_FRAC:
        suggested_boost = BASE_BOOST * max(1.0, TARGET_LONG_FRAC / long_frac)
    else:
        peak_boost = min(BASE_BOOST * (TARGET_LONG_FRAC / PEAK_FRAC), MAX_BOOST)
        suggested_boost = MIN_BOOST + (long_frac / PEAK_FRAC) * (peak_boost - MIN_BOOST)
    suggested_boost = round(min(suggested_boost, MAX_BOOST), 2)

    return {
        "n": n, "avg_len": avg_len, "max_len": max_len,
        "long_count": long_count, "long_frac": long_frac,
        "suggested_boost": suggested_boost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", required=True)
    ap.add_argument("--print_boost_only", action="store_true",
                     help="Print ONLY the suggested --length_boost number, for shell scripting")
    args = ap.parse_args()

    stats = analyze(args.train_ann)

    if args.print_boost_only:
        print(stats["suggested_boost"])
        return

    print(f"Word-length analysis: {args.train_ann}")
    print(f"  Samples: {stats['n']}")
    print(f"  Average length: {stats['avg_len']:.1f} chars")
    print(f"  Max length: {stats['max_len']} chars")
    print(f"  Words >= {LONG_THRESHOLD} chars: {stats['long_count']} ({100*stats['long_frac']:.1f}%)")
    print(f"  Suggested --length_boost: {stats['suggested_boost']}")
    print()
    print("  NOTE: heuristic based on Telugu/Tamil data points in this project so far —")
    print("  always check the length-bucketed WRR/CRR breakdown on actual eval results")
    print("  after training and adjust further if the gap isn't fully closed.")


if __name__ == "__main__":
    main()
