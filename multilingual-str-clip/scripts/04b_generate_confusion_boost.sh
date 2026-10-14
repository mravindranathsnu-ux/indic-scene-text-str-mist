#!/bin/bash
# Step 4b (OPTIONAL, auto-skips on a language's first-ever run): a SECOND,
# focused synthetic-data pass targeting the specific characters this
# language's model actually confuses -- learned from its own real test
# results (results/<language>_test_predictions.csv), using the same
# Levenshtein-alignment method behind every confusion-pair table in the
# project's error analysis report.
#
# Only makes sense once a language has been trained and evaluated at
# least once (round 2+) -- automatically SKIPS on a first-ever run, when
# no prior results exist yet to learn confusions from. No flag needed for
# that case; it just detects it.
#
# Runs AFTER 04_generate_synthetic_data.sh in the pipeline (needs
# TRAIN_AUGMENTED_ANN and TRAIN_IMAGE_ROOT already populated by that step,
# since this one APPENDS to them, not replaces them).
#
# Skip entirely with SKIP_CONFUSION_BOOST=1. Force regeneration (e.g.
# after a new eval result changes which characters are confusable) with
# FORCE_REGEN=1 -- the same flag 04 already uses, so one FORCE_REGEN=1
# forces a full fresh pass end to end.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 4b: Confusion-focused synthetic data boost for $LANGUAGE ==="
cd "$PROJECT_ROOT"

if [ "${SKIP_CONFUSION_BOOST:-0}" = "1" ]; then
    echo "SKIP_CONFUSION_BOOST=1 set — skipping this step entirely."
    exit 0
fi

PREDICTIONS_CSV="$PROJECT_ROOT/results/${LANGUAGE}_test_predictions.csv"
if [ ! -f "$PREDICTIONS_CSV" ]; then
    echo "No prior results found at $PREDICTIONS_CSV — nothing to learn confusions from yet."
    echo "(Expected on a language's first-ever run. Re-run the pipeline after this language has"
    echo "been trained and evaluated at least once, and this step will activate automatically.)"
    exit 0
fi

CONFUSION_OUT_DIR="$PROJECT_ROOT/data/$LANGUAGE/train/synthetic_confusion_boost"
FOCUS_WORDLIST="$PROJECT_ROOT/data/$LANGUAGE/train/confusion_focus_words.tsv"
TOP_N_CHARS="${CONFUSION_TOP_N_CHARS:-15}"
BOOST_MULTIPLIER="${CONFUSION_BOOST_MULTIPLIER:-3}"
MAX_WORDS="${CONFUSION_MAX_WORDS:-300}"
BOOST_SAMPLES=$(( SAMPLES_PER_WORD * BOOST_MULTIPLIER ))

echo "Learning confusable characters from: $PREDICTIONS_CSV"
"$PYTHON" generate_confusion_focus_wordlist.py \
    --predictions "$PREDICTIONS_CSV" \
    --train_ann "$TRAIN_ANN" \
    --out "$FOCUS_WORDLIST" \
    --out_format tsv \
    --top_n_chars "$TOP_N_CHARS" \
    --max_words "$MAX_WORDS"
WORDLIST_STATUS=$?
if [ $WORDLIST_STATUS -ne 0 ]; then
    echo "!! Could not build a confusion-focus word list (see above) — skipping the boost pass"
    echo "!! for this run. (Common cause: this language's result has no character-level"
    echo "!! confusions to learn from, e.g. very high accuracy already.)"
    exit 0
fi

EXISTING_TSV_LINES=0
EXISTING_IMAGE_COUNT=0
if [ -f "$CONFUSION_OUT_DIR/synthetic.tsv" ]; then
    EXISTING_TSV_LINES=$(wc -l < "$CONFUSION_OUT_DIR/synthetic.tsv" | tr -d ' ')
fi
if [ -d "$CONFUSION_OUT_DIR/images" ]; then
    EXISTING_IMAGE_COUNT=$(find "$CONFUSION_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" | wc -l | tr -d ' ')
fi

SKIP_RENDER=false
if [ "${FORCE_REGEN:-0}" = "1" ]; then
    echo "FORCE_REGEN=1 set — regenerating the confusion-boost pass even though it may already exist."
elif [ "$EXISTING_TSV_LINES" -gt 0 ] && [ "$EXISTING_TSV_LINES" = "$EXISTING_IMAGE_COUNT" ]; then
    echo "Found existing confusion-boost synthetic data: $EXISTING_TSV_LINES entries, "
    echo "$EXISTING_IMAGE_COUNT matching images in $CONFUSION_OUT_DIR/images/ — skipping"
    echo "(re-)generation. Set FORCE_REGEN=1 to regenerate anyway."
    SKIP_RENDER=true
elif [ "$EXISTING_TSV_LINES" -gt 0 ]; then
    echo "!! Found confusion-boost synthetic.tsv with $EXISTING_TSV_LINES entries but only"
    echo "!! $EXISTING_IMAGE_COUNT matching images — inconsistent (partial/interrupted previous"
    echo "!! run?). Regenerating from scratch to be safe."
fi

if [ "$SKIP_RENDER" = false ]; then
    echo "Generating confusion-focus synthetic data at ${BOOST_SAMPLES} samples/word"
    echo "(${BOOST_MULTIPLIER}x the normal --samples_per_word=${SAMPLES_PER_WORD}, targeting only the"
    echo "${TOP_N_CHARS} most-confused characters' words)..."
    "$PYTHON" generate_synthetic_data.py \
        --train_ann "$FOCUS_WORDLIST" \
        --fonts_dir "$FONTS_DIR" \
        --out_dir "$CONFUSION_OUT_DIR" \
        --samples_per_word "$BOOST_SAMPLES" \
        --exclude_ann "$VAL_ANN" "$TEST_ANN"
    if [ $? -ne 0 ]; then
        echo "!! Confusion-boost generation failed — train_augmented.tsv will NOT include it this run."
        exit 1
    fi
fi

echo "Linking confusion-boost images into $TRAIN_IMAGE_ROOT (flat, bare-filename convention)..."
mkdir -p "$TRAIN_IMAGE_ROOT"
find "$CONFUSION_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" -print0 \
    | xargs -0 ln -sf -t "$TRAIN_IMAGE_ROOT"

echo "Rebuilding $TRAIN_AUGMENTED_ANN from real + main synthetic + confusion-boost data..."
# REBUILD FRESH (overwrite, not append) from all three sources every time,
# mirroring how 04_generate_synthetic_data.sh itself always rebuilds
# train_augmented.tsv from scratch rather than appending. Appending here
# instead would duplicate this step's contribution every time the pipeline
# is re-run (caught directly while testing this script) -- rebuilding from
# scratch is what makes re-running the pipeline safe and idempotent.
cat "$TRAIN_ANN" \
    <(sed 's#^images/##' "$SYNTHETIC_OUT_DIR/synthetic.tsv") \
    <(sed 's#^images/##' "$CONFUSION_OUT_DIR/synthetic.tsv") \
    > "$TRAIN_AUGMENTED_ANN"

N_ADDED=$(wc -l < "$CONFUSION_OUT_DIR/synthetic.tsv" | tr -d ' ')
N_TOTAL=$(wc -l < "$TRAIN_AUGMENTED_ANN" | tr -d ' ')
BOOST_PCT=$(( 100 * N_ADDED / N_TOTAL ))
echo ""
echo "Confusion-boost share of total training data: $N_ADDED / $N_TOTAL (${BOOST_PCT}%)"
if [ "$BOOST_PCT" -gt 40 ]; then
    echo "!! WARNING: confusion-boost data is ${BOOST_PCT}% of the entire training set -- likely too"
    echo "!! dominant to be a 'focused supplement'. This exact situation destabilized a real training"
    echo "!! run (Telugu, round 2) before generate_confusion_focus_wordlist.py's --max_words cap was"
    echo "!! added. Consider lowering CONFUSION_TOP_N_CHARS, CONFUSION_BOOST_MULTIPLIER, or the focus"
    echo "!! word list's --max_words before training on this."
fi
echo ""
echo "train_augmented.tsv now includes $N_ADDED confusion-focused synthetic image(s), on top of"
echo "the real + main-synthetic data already in it."
if [ "$SKIP_RENDER" = true ]; then
    echo "(Images were not re-rendered this run — see above. Symlinks and train_augmented.tsv"
    echo "were refreshed regardless.)"
else
    echo ">>> ACTION REQUIRED: open a handful of images in $CONFUSION_OUT_DIR/images/ and visually"
    echo ">>> confirm they rendered correctly, same as the main synthetic step."
fi
