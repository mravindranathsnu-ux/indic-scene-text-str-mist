#!/bin/bash
# Step 9: Run inference on new images with the trained checkpoint.
# Usage: ./09_inference.sh /path/to/image_dir
#        (defaults to samples/<language>/ if no argument given)
# NOTE: this script's $1 is the image directory, not a language name (unlike
# the other numbered scripts) — set STR_LANGUAGE=<language> to pick which
# language's checkpoint/vocab to use, e.g.:
#     STR_LANGUAGE=malayalam ./09_inference.sh /path/to/images
#
# inference.py now prints every file it's about to use (checkpoint, vocab,
# image path/dir) with an explicit OK/MISSING check, and asks for
# confirmation before running. To skip the prompt, export INFER_YES=1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture $1 (image dir) BEFORE sourcing config.sh and clear positional
# params first, so config.sh's own "$1 = language" parsing (used by every
# other numbered script) doesn't misinterpret an image path as a language.
IMAGE_DIR_ARG="${1:-}"
set --
source "$SCRIPT_DIR/config.sh"

IMAGE_DIR="${IMAGE_DIR_ARG:-$SAMPLES_DIR}"

echo "=== Step 9: Inference on $IMAGE_DIR ==="
cd "$PROJECT_ROOT"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Image directory $IMAGE_DIR not found."
    echo "Usage: $0 /path/to/image_dir"
    exit 1
fi

# Catch the "empty of images" case here, before calling inference.py, so we
# can point at the obvious fallback (your real test images) instead of just
# reporting a generic empty-directory message — samples/<language>/ ships
# as an empty placeholder folder (just a README.txt) until you add your own
# images to it, which is easy to forget and looks identical to a bug.
SUPPORTED_EXTS="jpg jpeg png bmp tiff webp"
MATCH_COUNT=0
for ext in $SUPPORTED_EXTS; do
    COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -iname "*.$ext" 2>/dev/null | wc -l)
    MATCH_COUNT=$((MATCH_COUNT + COUNT))
done

if [ "$MATCH_COUNT" -eq 0 ]; then
    echo "!! No supported images found in $IMAGE_DIR."
    if [ -z "$IMAGE_DIR_ARG" ]; then
        echo ""
        echo "That's your test image folder (\$TEST_IMAGE_ROOT in config.sh) — it should"
        echo "contain your real $LANGUAGE test images. If it's genuinely empty, check that"
        echo "06_sanity_check.sh / 07_train.sh / 08_evaluate.sh ran against the data you"
        echo "expect, and that --test_image_root points at the right place."
        echo ""
        echo "To test on a DIFFERENT folder of images instead, pass it directly:"
        echo "    $0 /path/to/images"
    fi
    exit 1
fi

OUT_CSV="$RESULTS_DIR/${LANGUAGE}_inference_predictions.csv"
mkdir -p "$RESULTS_DIR"

EXTRA_FLAGS=()
if [ "${INFER_YES:-0}" = "1" ]; then
    EXTRA_FLAGS+=(--yes)
fi

"$PYTHON" inference.py \
    --ckpt "$OUT_DIR/best.pt" \
    --clip_ckpt "$CLIP_CKPT" \
    --image_dir "$IMAGE_DIR" \
    --decode beam --beam_size "$BEAM_SIZE" --length_penalty "$LENGTH_PENALTY" \
    --out_csv "$OUT_CSV" \
    "${EXTRA_FLAGS[@]}"
