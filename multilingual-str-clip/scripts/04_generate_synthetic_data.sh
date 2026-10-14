#!/bin/bash
# Step 4 (OPTIONAL but recommended once real-data hyperparameter tuning
# plateaus): Render synthetic word images to supplement real training
# data, then combine into train_augmented.tsv.
#
# Skips the (slow) rendering step automatically if synthetic data already
# exists and looks consistent with synthetic.tsv — safe to re-run this
# script (or the full pipeline) repeatedly without wasted regeneration
# time. Force a fresh regeneration (e.g. after changing --length_boost)
# with FORCE_REGEN=1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 4: Generate synthetic training data for $LANGUAGE ==="
cd "$PROJECT_ROOT"

EXISTING_TSV_LINES=0
EXISTING_IMAGE_COUNT=0
if [ -f "$SYNTHETIC_OUT_DIR/synthetic.tsv" ]; then
    EXISTING_TSV_LINES=$(wc -l < "$SYNTHETIC_OUT_DIR/synthetic.tsv" | tr -d ' ')
fi
if [ -d "$SYNTHETIC_OUT_DIR/images" ]; then
    EXISTING_IMAGE_COUNT=$(find "$SYNTHETIC_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" | wc -l | tr -d ' ')
fi

SKIP_RENDER=false
if [ "${FORCE_REGEN:-0}" = "1" ]; then
    echo "FORCE_REGEN=1 set — regenerating synthetic data even though it may already exist."
elif [ "$EXISTING_TSV_LINES" -gt 0 ] && [ "$EXISTING_TSV_LINES" = "$EXISTING_IMAGE_COUNT" ]; then
    echo "Found existing synthetic data: $EXISTING_TSV_LINES entries in synthetic.tsv, "
    echo "$EXISTING_IMAGE_COUNT matching images in $SYNTHETIC_OUT_DIR/images/ — skipping"
    echo "(re-)generation. Set FORCE_REGEN=1 to regenerate anyway (e.g. after changing"
    echo "--length_boost or --samples_per_word)."
    SKIP_RENDER=true
elif [ "$EXISTING_TSV_LINES" -gt 0 ]; then
    echo "!! Found synthetic.tsv with $EXISTING_TSV_LINES entries but only $EXISTING_IMAGE_COUNT"
    echo "!! matching images — inconsistent (partial/interrupted previous run?). Regenerating"
    echo "!! from scratch to be safe."
fi

if [ "$SKIP_RENDER" = false ]; then
    EXTRA_ARGS=()
    if [ -n "$CORPUS_FILE" ]; then
        EXTRA_ARGS+=(--corpus_file "$CORPUS_FILE")
    fi

    "$PYTHON" generate_synthetic_data.py \
        --train_ann "$TRAIN_ANN" \
        --fonts_dir "$FONTS_DIR" \
        --out_dir "$SYNTHETIC_OUT_DIR" \
        --samples_per_word "$SAMPLES_PER_WORD" \
        --length_boost "$LENGTH_BOOST" \
        --exclude_ann "$VAL_ANN" "$TEST_ANN" \
        "${EXTRA_ARGS[@]}"
fi

# Always re-run the symlink + combine step, even when rendering was
# skipped above — these are fast (no image rendering), and re-running them
# is what keeps things correct if train_image_root was freshly
# extracted/replaced since synthetic data was last generated (symlinks
# would otherwise be silently missing even though synthetic.tsv exists).
echo "Linking synthetic images into $TRAIN_IMAGE_ROOT (flat, bare-filename convention)..."
mkdir -p "$TRAIN_IMAGE_ROOT"
# NOT `ln -sf .../*.jpg "$TRAIN_IMAGE_ROOT/"` -- a shell glob expands to one
# argument per file, which hits Linux's ARG_MAX ("Argument list too long")
# once a language's synthetic set gets into the tens of thousands of images
# (hit directly during Tamil generation in this project). find+xargs batches
# the symlink calls automatically, so this scales to any dataset size.
find "$SYNTHETIC_OUT_DIR/images" -maxdepth 1 -iname "*.jpg" -print0 \
    | xargs -0 ln -sf -t "$TRAIN_IMAGE_ROOT"

echo "Combining real + synthetic annotations -> $TRAIN_AUGMENTED_ANN"
# generate_synthetic_data.py writes synthetic.tsv paths as "images/syn_XXXXXX.jpg"
# (relative to its OWN --out_dir) — strip that prefix so synthetic entries use
# the same bare-filename-relative-to-train_image_root convention as real.tsv,
# now that the images themselves are symlinked flat into train_image_root above.
# Mixing the two conventions unfixed is a real bug this project hit directly:
# train_image_root + "images/syn_X.jpg" duplicates the "images/" segment.
cat "$TRAIN_ANN" <(sed 's#^images/##' "$SYNTHETIC_OUT_DIR/synthetic.tsv") > "$TRAIN_AUGMENTED_ANN"

echo ""
if [ "$SKIP_RENDER" = true ]; then
    echo "(Synthetic images were not re-rendered — see above. Symlinks and"
    echo "train_augmented.tsv were refreshed regardless.)"
else
    echo ">>> ACTION REQUIRED: open a handful of images in $SYNTHETIC_OUT_DIR/images/"
    echo ">>> and visually confirm the script rendered correctly (garbled synthetic"
    echo ">>> labels would actively hurt training)."
fi
