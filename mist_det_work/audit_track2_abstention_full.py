#!/usr/bin/env python3
"""
audit_track2_abstention_full.py

Extends the earlier 15-crop manual spot-check of the fine-tuned VLM's
(Track 2's) MIST recognition output into a full-coverage audit across all
1,632 scored crops, plus a larger random stratified sample for direct
human reading.

WHY THIS EXISTS
The paper currently reports a manual inspection of 15 randomly-sampled
track2_text predictions (Section 6.3 / Limitations) that found no refusal
phrases, degenerate repeated-word loops, or unrelated fluent content --
consistent with the 0.0% abstention rate score_mist_baselines.py measures
for Track 2 (which, by design, never applies the abstention filter to
"track2" -- see ABSTENTION_APPLIES_TO in that script). 15 crops out of
1,632 is a reasonable spot-check but leaves open whether rare
refusal/hallucination behaviour exists elsewhere in the full set. This
script closes that gap two ways:

  1. FULL-COVERAGE AUTOMATED PASS: applies the EXACT SAME heuristic
     (is_abstention(), copied verbatim from score_mist_baselines.py --
     refusal-phrase list, >40-char length, >3-word count, 3+ repeated-word
     loop) to all 1,632 of Track 2's predictions, not just a sample. This
     tells us: if Track 2 *had* been filtered the way GPT-4V/Claude/
     Qwen3-VL-2B zero-shot are, how many of its 1,632 predictions would
     have been flagged, and why. Zero flagged = strong, full-coverage
     evidence for the paper's claim (not just a 15-crop proxy). Any
     flagged rows are printed in full so a human can read them directly
     and judge whether they're genuine abstention-like behaviour or a
     short prediction that just happens to cross one of these somewhat
     arbitrary thresholds (e.g. a genuinely long compound-word crop).

  2. EXPANDED RANDOM MANUAL SAMPLE: draws a larger (default 60, ~6/language)
     stratified random sample of track2_text predictions -- independent of
     whether the heuristic flags them -- for direct human reading. This is
     the thing the heuristic in (1) can't catch: a short (<=3 word, <=40
     char) prediction that is nevertheless a fluent hallucination unrelated
     to the crop, mirroring the failure mode Claude showed. Printed with
     gcv_text alongside for context, same format as the original 15-crop
     check so it's a straightforward "read these lines" task.

Neither step requires the GPU or the base model -- this only reads the
already-written manifest.csv, so it runs in seconds in any Python env with
no extra dependencies.

Usage:
    cd ~/mist_det_work        (or wherever manifest.csv lives)
    python3 audit_track2_abstention_full.py
    # or, if the manifest isn't at the default path:
    python3 audit_track2_abstention_full.py --manifest /path/to/manifest.csv
    # to change the size of the expanded random manual sample:
    python3 audit_track2_abstention_full.py --sample-size 60 --seed 42
"""
import argparse
import csv
import os
import random


DEFAULT_MANIFEST = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")

# --- copied verbatim from score_mist_baselines.py so this audit uses the
# exact same criteria the paper already describes, not a redefinition. ---
ABSTENTION_LEN_THRESHOLD = 40
REFUSAL_PHRASES = [
    "unable to", "i cannot", "i can't", "i'm sorry", "i am sorry",
    "cannot extract", "can't extract", "not able to", "please provide",
    "could you provide", "can you provide", "can you describe",
    "as an ai", "i don't have the ability", "i do not have the ability",
    "sorry",
]
ABSTENTION_WORD_THRESHOLD = 3


def is_abstention(text):
    if not text:
        return False
    if len(text) > ABSTENTION_LEN_THRESHOLD:
        return True
    words = text.split()
    if len(words) > ABSTENTION_WORD_THRESHOLD:
        return True
    if len(words) >= 3 and len(set(w.lower() for w in words)) == 1:
        return True
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def flag_reason(text):
    """Same predicate as is_abstention(), but reports WHICH rule fired,
    for the printed breakdown below."""
    if not text:
        return None
    if len(text) > ABSTENTION_LEN_THRESHOLD:
        return f"length>{ABSTENTION_LEN_THRESHOLD} ({len(text)} chars)"
    words = text.split()
    if len(words) > ABSTENTION_WORD_THRESHOLD:
        return f"word-count>{ABSTENTION_WORD_THRESHOLD} ({len(words)} words)"
    if len(words) >= 3 and len(set(w.lower() for w in words)) == 1:
        return "repeated-word loop"
    lower = text.lower()
    matched = [p for p in REFUSAL_PHRASES if p in lower]
    if matched:
        return f"refusal phrase ({matched[0]!r})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--sample-size", type=int, default=60,
                     help="Size of the expanded random manual-inspection sample (default 60, "
                          "~6 per language across the 10 languages).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"No manifest at {args.manifest}")

    with open(args.manifest, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    if "track2_text" not in rows[0]:
        raise SystemExit("No track2_text column in manifest -- run run_track2_vlm.py first.")

    # Same scoring universe as score_mist_baselines.py: rows where GCV
    # itself returned pseudo-GT text, and Track 2 actually produced a
    # non-exception prediction for that crop.
    scored = [r for r in rows if r.get("status") == "ok" and r.get("track2_status") == "ok"]
    print(f"Scoring universe: {len(scored)} crops (GCV pseudo-GT available AND Track 2 produced a prediction).\n")

    # ---------- Step 1: full-coverage automated pass over all 1,632 ----------
    flagged = []
    for r in scored:
        reason = flag_reason(r["track2_text"])
        if reason:
            flagged.append((r, reason))

    print("=" * 78)
    print(f"STEP 1: full-coverage automated scan, all {len(scored)} crops")
    print("=" * 78)
    print(f"Flagged by the same heuristic used for GPT-4V/Claude/Qwen3-VL-2B zero-shot: "
          f"{len(flagged)}/{len(scored)} ({100*len(flagged)/len(scored):.2f}%)\n")

    if flagged:
        print(f"{'language':<12s}{'reason':<28s}{'gcv_text':<20s}track2_text")
        print("-" * 90)
        for r, reason in flagged:
            print(f"{r['language']:<12s}{reason:<28s}{r['gcv_text'][:18]:<20s}{r['track2_text']}")
        print(f"\n^^ {len(flagged)} row(s) above matched the abstention heuristic -- read them directly: "
              f"a short prediction that just happens to be a long compound word (false positive) "
              f"looks very different from an actual refusal or unrelated hallucination (true positive).")
    else:
        print("No crops flagged. Zero of the 1,632 predictions match the refusal-phrase, length, "
              "word-count, or repeated-word-loop criteria -- full-coverage confirmation of the "
              "0.0% measured abstention rate, not just the earlier 15-crop sample.")

    # ---------- Step 2: expanded random stratified manual sample ----------
    print("\n" + "=" * 78)
    print(f"STEP 2: expanded random manual-inspection sample (n={args.sample_size}, seed={args.seed})")
    print("=" * 78)

    rng = random.Random(args.seed)
    languages = sorted(set(r["language"] for r in scored))
    per_lang = max(1, args.sample_size // len(languages))
    sample = []
    for lang in languages:
        lang_rows = [r for r in scored if r["language"] == lang]
        k = min(per_lang, len(lang_rows))
        sample.extend(rng.sample(lang_rows, k))
    # top up to exactly sample-size (or as close as possible) from the remainder
    remaining = [r for r in scored if r not in sample]
    shortfall = args.sample_size - len(sample)
    if shortfall > 0 and remaining:
        sample.extend(rng.sample(remaining, min(shortfall, len(remaining))))
    rng.shuffle(sample)

    print(f"{len(sample)} predictions sampled across {len(languages)} languages "
          f"(~{per_lang}/language). Read each track2_text below and judge: is this a plausible "
          f"attempted transcription (right or wrong), or a refusal / unrelated fluent hallucination?\n")
    for r in sample:
        gcv = r["gcv_text"] if r["gcv_text"] else "(empty -- GCV found text but it's blank in this field)"
        print(f"{r['language']:<12s} gcv={gcv!r:<22s} track2={r['track2_text']!r}")

    print(f"\nDone. {len(flagged)}/{len(scored)} auto-flagged (Step 1), "
          f"{len(sample)} additional rows for manual reading (Step 2).")
    print("Paste this entire output back to fold the extended audit into the paper.")


if __name__ == "__main__":
    main()
