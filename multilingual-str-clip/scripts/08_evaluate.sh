#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./08_evaluate.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 8: Evaluate the trained checkpoint on the held-out test set —
# reports WRR/CRR (and CER/WER), and writes per-sample predictions to CSV.
#
# evaluate.py now prints every file it's about to use (checkpoint, vocab,
# test data) with an explicit OK/MISSING check, spot-checks that test
# images actually resolve, and asks for confirmation before running — run
# this from an actual terminal. To skip the prompt, export EVAL_YES=1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 8: Evaluate $LANGUAGE model ==="
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"

EXTRA_FLAGS=()
if [ "${EVAL_YES:-0}" = "1" ]; then
    EXTRA_FLAGS+=(--yes)
fi

"$PYTHON" evaluate.py \
    --ckpt "$OUT_DIR/best.pt" \
    --clip_ckpt "$CLIP_CKPT" \
    --test_ann "$TEST_ANN" \
    --test_image_root "$TEST_IMAGE_ROOT" \
    --decode beam --beam_size "$BEAM_SIZE" --length_penalty "$LENGTH_PENALTY" \
    --out_csv "$RESULTS_DIR/${LANGUAGE}_test_predictions.csv" \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "Per-sample predictions: $RESULTS_DIR/${LANGUAGE}_test_predictions.csv"
