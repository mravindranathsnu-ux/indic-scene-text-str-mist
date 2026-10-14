#!/usr/bin/env python3
"""
score_mist_baselines.py

Scores every recognition baseline in manifest.csv (Tesseract,
PaddleOCR, EasyOCR, GPT-4V, Claude, Track 1 CLIP) against the Google
Vision pseudo-ground-truth (gcv_text), producing a BSTD-Table-9-style
WRR/CRR comparison -- overall and broken down per language.

Metric definitions match multilingual-str-clip's own utils.py exactly
(editdistance-based CER/WER, CRR = 1-CER, WRR = exact-string-match
accuracy) so these numbers are directly comparable to the WRR/CRR
already reported for Phase 1's own test sets.

Scoring universe and conventions (per decisions made earlier this
project):
  - Only rows where Google Vision itself returned text (its own
    "status" column == "ok", ~1632/3000 rows) are scored -- rows GCV
    couldn't read have no pseudo-GT to compare against. That ~46% GCV
    couldn't read is itself a reported finding (crop-size/legibility
    floor), not something to paper over here.
  - Google Vision itself is NOT scored as a system (it's the pseudo-
    GT source -- scoring it against itself would be circular, same
    reasoning as BSTD's Table 9 never re-scoring its own reference).
  - Per system, rows marked "unsupported_language" for that engine
    (PaddleOCR/EasyOCR only cover some Indic scripts) are EXCLUDED
    from that system's denominator -- "NA", matching BSTD's own
    convention, not counted as a wrong answer.
  - Per system, rows marked "no_text_detected" or "exception:..."
    ARE included and scored as a wrong prediction (empty string vs.
    the real gcv_text) -- these are genuine misses by a system that
    was actually run on that language, not "not applicable".

Reports BOTH an exact-string-match score (matching the project's own
established WRR convention, for apples-to-apples comparison with
Phase 1's already-reported numbers) and a normalized one (stripped +
casefolded) -- OCR outputs from six very different systems won't
agree on casing/whitespace even when "correct", so exact-match alone
likely understates every system except the one trained to match GCV's
own conventions. Both are reported rather than silently picking one.

ABSTENTION HANDLING (added after inspecting real GPT-4V/Claude output,
Sep 2026): on small/blurry non-Latin-script crops, GPT-4V frequently
returns a boilerplate refusal ("I'm unable to extract text from
images that use non-Latin characters...") instead of attempting OCR,
and Claude frequently returns fluent but entirely UNRELATED
hallucinated text (an e-commerce parts catalog, a physics
explanation, a National Park description -- none related to the
actual crop) rather than declining. Both are real, reportable
findings about these systems' behavior on this dataset -- but scoring
either one character-by-character against a 5-char reference word
produces meaningless numbers (CER > 100%, negative CRR) that obscure
the finding rather than measure it.

So: any prediction matching a refusal phrase, OR longer than
ABSTENTION_LEN_THRESHOLD characters (the real gcv_text reference
max, observed across all 1632 scored rows, is 36 chars -- 40 is
comfortably above any genuine single/multi-word crop label and below
every observed refusal/hallucination), is flagged as an ABSTENTION:
excluded from WRR/CER/WER (so those numbers reflect only genuine
attempts) and counted separately as its own rate per system/language.
This is not free-form judgment call territory -- it's a length +
phrase heuristic tuned against this dataset's actual observed
distribution, documented here so it can be checked/adjusted.

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    pip install editdistance   # if not already in this env
    python3 score_mist_baselines.py
    # writes:
    #   recognition_gt/scores_overall.csv       (one row per system x match-type)
    #   recognition_gt/scores_by_language.csv   (one row per system x language x match-type)
    # and prints both as tables, plus a BSTD-Table-9-style WRR%% grid.
"""
import csv
import os

try:
    import editdistance
except ImportError:
    raise SystemExit(
        "Missing dependency: editdistance. Install with:\n"
        "  pip install editdistance\n"
        "(multilingual-str-clip depends on this too, so it may already be "
        "installed there -- but mist_det is a separate conda env, so check "
        "you're not assuming it carried over.)"
    )

MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")
OUT_DIR = os.path.expanduser("~/mist_det_work/recognition_gt")

# key, text_column, status_column, display_name
SYSTEMS = [
    ("tesseract", "tesseract_text", "tesseract_status", "Tesseract"),
    ("paddleocr", "paddleocr_text", "paddleocr_status", "PaddleOCR"),
    ("easyocr",   "easyocr_text",   "easyocr_status",   "EasyOCR"),
    ("gpt4v",     "gpt4v_text",     "gpt4v_status",      "GPT-4V"),
    ("claude",    "claude_text",    "claude_status",     "Claude"),
    ("clip",      "clip_text",      "clip_status",       "Track1-CLIP"),
    ("qwen3vl",   "qwen3vl_text",   "qwen3vl_status",    "Qwen3-VL-2B"),
    ("track2",    "track2_text",    "track2_status",     "Track2-VLM-FT"),
]

NA_STATUSES = {"unsupported_language"}  # excluded from that system's denominator

# Abstention detection only applies to the general-purpose VLM systems.
# Dedicated OCR engines (Tesseract/PaddleOCR/EasyOCR/Track1-CLIP) don't
# "refuse" or "hallucinate unrelated prose" the way a general-purpose
# VLM does -- a garbled Tesseract output like '૧૪, 7૧૪૧૪૪૪૪૧૪ ;\n«ર
# /-«.#.#' is a genuine (if wrong) OCR attempt, exactly what CER is
# designed to measure, not a non-answer. Applying the abstention
# filter to those engines too would misclassify their real failures
# as excludable "NA-like" cases and artificially flatter their scores.
#
# Qwen3-VL-2B (added Sep 2026) shows the same failure family as
# GPT-4V/Claude, confirmed by inspecting a 20-crop sample before the
# full run: on crops where GCV itself found no text, it frequently
# fabricates plausible-looking content instead of a genuine attempt --
# a phrase repeated 3x ("A H A T A R A" x3), a fabricated numbered
# sequence ("1 2 3 4 5 ... 18"), and fluent unrelated English
# ("THE TALK ABOUT THE CITY") -- the same word-count/repetition
# pattern the heuristic below was built to catch for GPT-4V/Claude,
# so it generalizes directly with no new heuristics needed.
#
# Track2-VLM-FT (added after run_track2_vlm.py) is deliberately NOT included
# here yet. It's fine-tuned specifically as a single-word OCR model (unlike
# the general-purpose zero-shot VLMs above), so it may not exhibit the same
# refusal/hallucination failure mode at all -- inspect a real sample of its
# track2_text output on crops where gcv itself found little/nothing before
# deciding whether to add "track2" to this set, the same way qwen3vl's
# membership here was confirmed by inspection rather than assumed.
ABSTENTION_APPLIES_TO = {"gpt4v", "claude", "qwen3vl"}

# See ABSTENTION HANDLING note above: length threshold is set just above the
# real observed max gcv_text reference length (36 chars across all 1632
# scored rows) -- comfortably beyond any genuine crop label, comfortably
# below the refusal/hallucination responses actually observed.
ABSTENTION_LEN_THRESHOLD = 40
REFUSAL_PHRASES = [
    "unable to", "i cannot", "i can't", "i'm sorry", "i am sorry",
    "cannot extract", "can't extract", "not able to", "please provide",
    "could you provide", "can you provide", "can you describe",
    "as an ai", "i don't have the ability", "i do not have the ability",
    # "sorry" alone, not just "i'm sorry": run_vlm_baselines.py's
    # parse_list_response has a real bug where its single-quote regex
    # fallback (meant to pull ['word1','word2']-style quoted items out
    # of a clean list response) misfires on ordinary English
    # contractions in a refusal sentence -- "I'm sorry, but I can't..."
    # gets mis-sliced into "m sorry, but I can" (the text between the
    # apostrophe in "I'm" and the one in "can't"), which is short enough
    # to slip past the length threshold above. Catching "sorry" alone
    # closes that gap; a real crop legitimately OCR'ing to "sorry" is
    # vanishingly unlikely on this dataset (street signs / product
    # labels in 10 Indian languages).
    "sorry",
]


# 92% of gcv_text references are a single word, 98.6% are 1-2 words
# (checked across all 1632 scored rows: max is 7 words, only 8 rows
# have 5+). A prediction with more than 3 words on what's meant to be
# a single cropped word is almost never a genuine transcription
# attempt -- inspecting real Claude output turned up exactly this
# pattern: unprompted transliteration/translation glosses ("सवारी
# सवारी savari sawaari ride vehicle" for a crop reading "वाघचव") and
# degenerate repeated-word loops ("research research research
# research", "LULLABY LULLABY LULLABY") rather than refusals or
# plausible short guesses.
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
        return True  # same word repeated 3+ times -- degenerate loop, not a real guess
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def compute_cer(preds, refs):
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        total_dist += editdistance.eval(p, r)
        total_len += max(len(r), 1)
    return total_dist / max(total_len, 1)


def compute_wer(preds, refs):
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        p_words, r_words = p.split(), r.split()
        total_dist += editdistance.eval(p_words, r_words)
        total_len += max(len(r_words), 1)
    return total_dist / max(total_len, 1)


def compute_wrr(preds, refs):
    if not refs:
        return 0.0
    correct = sum(1 for p, r in zip(preds, refs) if p == r)
    return correct / len(refs)


def compute_all_metrics(preds, refs):
    cer = compute_cer(preds, refs)
    return {
        "CER": cer,
        "WER": compute_wer(preds, refs),
        "CRR": 1.0 - cer,
        "WRR": compute_wrr(preds, refs),
        "num_samples": len(refs),
    }


def normalize(s):
    return " ".join((s or "").strip().split())


def score_subset(rows, key, text_col, status_col, normalized=False):
    """Returns None if there's nothing to score, else a dict with the
    abstention rate (over ALL non-NA rows) plus WRR/CRR/WER/CER computed
    only over the non-abstained ("attempted") subset. Abstention
    filtering only applies to systems in ABSTENTION_APPLIES_TO -- see
    that constant's comment."""
    scored = [r for r in rows if r[status_col] not in NA_STATUSES]
    if not scored:
        return None
    n_total = len(scored)
    if key in ABSTENTION_APPLIES_TO:
        attempted = [r for r in scored if not is_abstention(r[text_col])]
    else:
        attempted = scored
    n_abstained = n_total - len(attempted)

    if not attempted:
        return {"num_samples": n_total, "num_abstained": n_abstained,
                "abstention_rate": 1.0, "WRR": 0.0, "CRR": 0.0, "WER": 1.0, "CER": 1.0}

    preds = [r[text_col] for r in attempted]
    refs = [r["gcv_text"] for r in attempted]
    if normalized:
        preds = [normalize(p).casefold() for p in preds]
        refs = [normalize(r).casefold() for r in refs]
    m = compute_all_metrics(preds, refs)
    m["num_samples"] = n_total
    m["num_abstained"] = n_abstained
    m["abstention_rate"] = n_abstained / n_total
    return m


def main():
    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    scored_universe = [r for r in rows if r["status"] == "ok"]
    print(f"Scoring universe: {len(scored_universe)}/{len(rows)} rows "
          f"(Google Vision pseudo-GT available).\n")

    languages = sorted(set(r["language"] for r in scored_universe))
    available_systems = [s for s in SYSTEMS if s[1] in rows[0]]
    skipped = [s[3] for s in SYSTEMS if s[1] not in rows[0]]
    if skipped:
        print(f"[note] no column yet for: {', '.join(skipped)} (not run) -- skipping.\n")

    overall_rows, by_lang_rows = [], []

    for key, text_col, status_col, display in available_systems:
        for normalized in (False, True):
            m = score_subset(scored_universe, key, text_col, status_col, normalized)
            if m:
                overall_rows.append({"system": display, "match": "normalized" if normalized else "exact", **m})

        for lang in languages:
            lang_rows = [r for r in scored_universe if r["language"] == lang]
            for normalized in (False, True):
                m = score_subset(lang_rows, key, text_col, status_col, normalized)
                if m:
                    by_lang_rows.append({
                        "system": display, "language": lang,
                        "match": "normalized" if normalized else "exact", **m,
                    })

    fields = ["system", "match", "num_samples", "num_abstained", "abstention_rate", "WRR", "CRR", "WER", "CER"]
    os.makedirs(OUT_DIR, exist_ok=True)
    overall_csv = os.path.join(OUT_DIR, "scores_overall.csv")
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overall_rows)

    by_lang_csv = os.path.join(OUT_DIR, "scores_by_language.csv")
    with open(by_lang_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "language"] + fields[1:])
        writer.writeheader()
        writer.writerows(by_lang_rows)

    print(f"{'system':<14s}{'match':<12s}{'N':>6s}{'abstain':>9s}{'WRR':>9s}{'CRR':>9s}{'WER':>9s}{'CER':>9s}")
    print(f"  (WRR/CRR/WER/CER computed only over non-abstained attempts; 'abstain' = refusal/hallucination rate, see docstring)")
    print("-" * 77)
    for r in overall_rows:
        print(f"{r['system']:<14s}{r['match']:<12s}{r['num_samples']:>6d}"
              f"{r['abstention_rate']*100:>8.1f}%{r['WRR']*100:>8.2f}%{r['CRR']*100:>8.2f}%"
              f"{r['WER']*100:>8.2f}%{r['CER']*100:>8.2f}%")

    print("\n\nWRR%% by language, non-abstained attempts only (exact match) -- BSTD Table 9 style\n")
    header = f"{'system':<14s}" + "".join(f"{l[:7]:>9s}" for l in languages)
    print(header)
    for key, text_col, status_col, display in available_systems:
        cells = []
        for lang in languages:
            match = next((r for r in by_lang_rows
                          if r["system"] == display and r["language"] == lang and r["match"] == "exact"), None)
            cells.append(f"{match['WRR']*100:>8.1f}" if match else f"{'NA':>8s}")
        print(f"{display:<14s}" + "".join(f"{c:>9s}" for c in cells))

    print("\n\nAbstention rate %% by language (exact match) -- refusal/hallucination frequency\n")
    print(header)
    for key, text_col, status_col, display in available_systems:
        cells = []
        for lang in languages:
            match = next((r for r in by_lang_rows
                          if r["system"] == display and r["language"] == lang and r["match"] == "exact"), None)
            cells.append(f"{match['abstention_rate']*100:>8.1f}" if match else f"{'NA':>8s}")
        print(f"{display:<14s}" + "".join(f"{c:>9s}" for c in cells))

    print(f"\nWrote: {overall_csv}")
    print(f"Wrote: {by_lang_csv}")


if __name__ == "__main__":
    main()
