#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./06_sanity_check.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 6: Validate the pipeline BEFORE a full training run — data
# checks, preprocessing visualization, and tiny-batch overfit tests for
# both Stage 1 (frozen backbone) and Stage 2 (unfrozen backbone). Catches
# most pipeline bugs and bad hyperparameters in minutes instead of after
# a multi-hour training run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 6: Sanity check pipeline for $LANGUAGE ==="
cd "$PROJECT_ROOT"

"$PYTHON" sanity_check.py \
    --train_ann "$TRAIN_ANN" \
    --train_image_root "$TRAIN_IMAGE_ROOT" \
    --clip_ckpt "$CLIP_CKPT" \
    --test_ann "$TEST_ANN" \
    --vocab_path "$VOCAB_PATH" \
    --overfit_steps "$OVERFIT_STEPS" \
    --overfit_lr "$OVERFIT_LR" \
    --overfit_unfreeze_last_n "$UNFREEZE_LAST_N" \
    --overfit_backbone_lr "$BACKBONE_LR"

echo ""
echo ">>> Review the OVERALL VERDICT above before proceeding to step 07 (train)."
