#!/bin/bash
# Step 4c (OPTIONAL, auto-skips on a language's first-ever run): a THIRD
# synthetic-data pass generating minimal-pair contrastive words for the
# specific character PAIRS this language's model confuses (learned from
# its own real test results) -- complementary to 04b, which boosts real
# words containing a hard character generically. This instead constructs
# synthetic "twin" words by substituting one confused character for its
# pair partner throughout a real word, giving the model direct, paired
# exposure to both glyphs in otherwise-identical visual contexts. The twin
# need not be a real dictionary word -- it's still a valid image-label
# training pair, since the rendered image shows exactly the substituted
# characters.
#
# Runs AFTER 04b in the pipeline (needs TRAIN_AUGMENTED_ANN and
# TRAIN_IMAGE_ROOT already populated by 04 and 04b, since this appends to
# them, not replaces them).
#
# Skip entirely with SKIP_CONFUSION_PAIR_BOOST=1. Force regeneration with
# FORCE_REGEN=1 (same flag 04/04b already use).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 4c: Minimal-pair confusion synthetic boost for $LANGUAGE ==="
cd "$PROJECT_ROOT"

if [ "${SKIP_CONFUSION_PAIR_BOOST:-0}" = "1" ]; then
    echo "SKIP_CONFUSION_PAIR_BOOST=1 set — skipping this step entirely."
    exit 0
fi

PREDICTIONS_CSV="$PROJECT_ROOT/results/${LANGUAGE}_test_predictions.csv"
if [ ! -f "$PREDICTIONS_CSV" ]; then
    echo "No prior results found at $PREDICTIONS_CSV — nothing to learn confusion pairs from yet."
    echo "(Expected on a language's first-ever run. Activates automatically from round 2 on.)"
    exit 0
fi

PAIR_OUT_DIR="$PROJECT_ROOT/data/$LANGUAGE/train/synthetic_confusion_pairs"
PAIR_WORDLIST="$PROJECT_ROOT/data/$LANGUAGE/train/confusion_pair_words.tsv"
TOP_N_PAIRS="${CONFUSION_TOP_N_PAIRS:-10}"
MAX_WORDS_PER_PAIR="${CONFUSION_MAX_WORDS_PER_PAIR:-30}"
BOOST_MULTIPLIER="${CONFUSION_PAIR_BOOST_MULTIPLIER:-3}"
BOOST_SAMPLES=$(( SAMPLES_PER_WORD * BOOST_MULTIPLIER ))

echo "Learning confusion pairs from: $PREDICTIONS_CSV"
"$PYTHON" generate_confusion_pair_wordlist.py \
    --predictions "$PREDICTIONS_CSV" \
    --train_ann "$TRAIN_ANN" \
    --out "$PAIR_WORDLIST" \
    --out_format tsv \
    --top_n_pairs "$TOP_N_PAIRS" \
    --max_words_per_pair "$MAX_WORDS_PER_PAIR"
WORDLIST_STATUS=$?
if [ $WORDLIST_STATUS -ne 0 ]; then
    echo "!! Could not build a confusion-pair word list (see above) — skipping this boost pass"
    echo "!! for this run. (Common cause: no character-level confusion pairs to learn from.)"
    exit 0
fi

EXISTING_TSV_LINES=0
EXISTING_IMAGE_COUNT=0
if [ -f "$PAIR_OUT_DIR/synthetic.tsv" ]; then
    EXISTING_TSV_LINES=$(wc -l < "$PAIR_OUT_DIR/synthetic.tsv" | tr -d ' ')
fi
if [ -d "$PAIR_OUT_DIR/images" ]; then
    EXISTING_IMAGE_COUNT=$(find "$PAIR_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" | wc -l | tr -d ' ')
fi

SKIP_RENDER=false
if [ "${FORCE_REGEN:-0}" = "1" ]; then
    echo "FORCE_REGEN=1 set — regenerating the confusion-pair boost pass even though it may already exist."
elif [ "$EXISTING_TSV_LINES" -gt 0 ] && [ "$EXISTING_TSV_LINES" = "$EXISTING_IMAGE_COUNT" ]; then
    echo "Found existing confusion-pair synthetic data: $EXISTING_TSV_LINES entries, "
    echo "$EXISTING_IMAGE_COUNT matching images in $PAIR_OUT_DIR/images/ — skipping"
    echo "(re-)generation. Set FORCE_REGEN=1 to regenerate anyway."
    SKIP_RENDER=true
elif [ "$EXISTING_TSV_LINES" -gt 0 ]; then
    echo "!! Found confusion-pair synthetic.tsv with $EXISTING_TSV_LINES entries but only"
    echo "!! $EXISTING_IMAGE_COUNT matching images — inconsistent (partial/interrupted previous"
    echo "!! run?). Regenerating from scratch to be safe."
fi

if [ "$SKIP_RENDER" = false ]; then
    echo "Generating confusion-pair synthetic data at ${BOOST_SAMPLES} samples/word"
    echo "(${BOOST_MULTIPLIER}x the normal --samples_per_word=${SAMPLES_PER_WORD}, targeting minimal-pair"
    echo "twins for the top ${TOP_N_PAIRS} confusion pairs, up to ${MAX_WORDS_PER_PAIR} words each)..."
    "$PYTHON" generate_synthetic_data.py \
        --train_ann "$PAIR_WORDLIST" \
        --fonts_dir "$FONTS_DIR" \
        --out_dir "$PAIR_OUT_DIR" \
        --samples_per_word "$BOOST_SAMPLES" \
        --exclude_ann "$VAL_ANN" "$TEST_ANN"
    if [ $? -ne 0 ]; then
        echo "!! Confusion-pair boost generation failed — train_augmented.tsv will NOT include it this run."
        exit 1
    fi
fi

echo "Linking confusion-pair images into $TRAIN_IMAGE_ROOT (flat, bare-filename convention)..."
mkdir -p "$TRAIN_IMAGE_ROOT"
find "$PAIR_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" -print0 \
    | xargs -0 ln -sf -t "$TRAIN_IMAGE_ROOT"

echo "Rebuilding $TRAIN_AUGMENTED_ANN from real + main synthetic + confusion-boost + confusion-pair data..."
# Same idempotency lesson as 04b: rebuild FRESH (overwrite) from every
# source every time, never append -- appending would duplicate this step's
# contribution on every pipeline re-run (caught directly while testing 04b;
# same fix applied here from the start).
CONFUSION_OUT_DIR="$PROJECT_ROOT/data/$LANGUAGE/train/synthetic_confusion_boost"
if [ -f "$CONFUSION_OUT_DIR/synthetic.tsv" ]; then
    cat "$TRAIN_ANN" \
        <(sed 's#^images/##' "$SYNTHETIC_OUT_DIR/synthetic.tsv") \
        <(sed 's#^images/##' "$CONFUSION_OUT_DIR/synthetic.tsv") \
        <(sed 's#^images/##' "$PAIR_OUT_DIR/synthetic.tsv") \
        > "$TRAIN_AUGMENTED_ANN"
else
    # 04b hasn't run yet this session (e.g. SKIP_CONFUSION_BOOST=1 was set) --
    # rebuild from real + main synthetic + this step only, so it's still
    # correct rather than assuming a file that isn't there.
    cat "$TRAIN_ANN" \
        <(sed 's#^images/##' "$SYNTHETIC_OUT_DIR/synthetic.tsv") \
        <(sed 's#^images/##' "$PAIR_OUT_DIR/synthetic.tsv") \
        > "$TRAIN_AUGMENTED_ANN"
fi

N_ADDED=$(wc -l < "$PAIR_OUT_DIR/synthetic.tsv" | tr -d ' ')
N_TOTAL=$(wc -l < "$TRAIN_AUGMENTED_ANN" | tr -d ' ')
BOOST_PCT=$(( 100 * N_ADDED / N_TOTAL ))
echo ""
echo "Confusion-pair share of total training data: $N_ADDED / $N_TOTAL (${BOOST_PCT}%)"
if [ "$BOOST_PCT" -gt 40 ]; then
    echo "!! WARNING: confusion-pair data is ${BOOST_PCT}% of the entire training set -- likely too"
    echo "!! dominant to be a 'focused supplement'. Consider lowering CONFUSION_TOP_N_PAIRS,"
    echo "!! CONFUSION_MAX_WORDS_PER_PAIR, or CONFUSION_PAIR_BOOST_MULTIPLIER."
fi

echo ""
echo "train_augmented.tsv now includes $N_ADDED confusion-pair minimal-pair image(s), on top of"
echo "the real + main-synthetic + confusion-focus data already in it."
if [ "$SKIP_RENDER" = true ]; then
    echo "(Images were not re-rendered this run — see above. Symlinks and train_augmented.tsv"
    echo "were refreshed regardless.)"
else
    echo ">>> ACTION REQUIRED: open a handful of images in $PAIR_OUT_DIR/images/ and visually"
    echo ">>> confirm they rendered correctly, same as the other synthetic steps."
fi
