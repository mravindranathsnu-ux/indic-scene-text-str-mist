#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./07_train.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 7: Train. Uses train_augmented.tsv automatically if step 04 was
# run (falls back to plain train.tsv otherwise).
#
# train.py now prints every parameter and asks for interactive
# confirmation before starting — run this from an actual terminal (not
# piped/redirected). To skip the prompt (e.g. for scripted/unattended
# runs), export TRAIN_YES=1 before calling this script.
#
# To resume a force-stopped/interrupted run instead of starting over from
# epoch 1: export TRAIN_RESUME=1 before calling this script (requires
# <out_dir>/resume_state.pt from the interrupted run — saved automatically
# every epoch, so any run stopped after epoch 1 completed can be resumed).
# If neither TRAIN_RESUME nor --resume is given but resume_state.pt exists,
# train.py now prompts interactively (defaulting to RESUME, not overwrite)
# instead of silently starting fresh — TRAIN_YES=1 also answers that prompt.
#
# To roll back further than the latest epoch (e.g. a periodic snapshot from
# --checkpoint_every, default every 5 epochs, saved as
# <out_dir>/checkpoint_epoch_<N>.pt and never overwritten): export
# TRAIN_RESUME_FROM=<path to that file> before calling this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 7: Train $LANGUAGE model ==="
cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

TRAIN_INPUT="$TRAIN_ANN"
if [ -f "$TRAIN_AUGMENTED_ANN" ]; then
    TRAIN_INPUT="$TRAIN_AUGMENTED_ANN"
    echo "Using augmented training data: $TRAIN_INPUT"
else
    echo "Using real-only training data: $TRAIN_INPUT (run 04_generate_synthetic_data.sh first for more data)"
fi

EXTRA_FLAGS=()
if [ "${TRAIN_YES:-0}" = "1" ]; then
    EXTRA_FLAGS+=(--yes)
fi
if [ "${TRAIN_RESUME:-0}" = "1" ]; then
    EXTRA_FLAGS+=(--resume)
fi
if [ -n "${TRAIN_RESUME_FROM:-}" ]; then
    EXTRA_FLAGS+=(--resume_from "$TRAIN_RESUME_FROM")
fi

"$PYTHON" train.py \
    --train_ann "$TRAIN_INPUT" \
    --val_ann "$VAL_ANN" \
    --train_image_root "$TRAIN_IMAGE_ROOT" \
    --val_image_root "$VAL_IMAGE_ROOT" \
    --clip_ckpt "$CLIP_CKPT" \
    --language "$LANGUAGE" \
    --vocab_path "$VOCAB_PATH" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --warmup_frozen_epochs "$WARMUP_FROZEN_EPOCHS" \
    --unfreeze_last_n "$UNFREEZE_LAST_N" \
    --lr "$LR" \
    --backbone_lr "$BACKBONE_LR" \
    --checkpoint_every "${CHECKPOINT_EVERY:-5}" \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "Training complete. Checkpoints: $OUT_DIR/{best.pt,last.pt}"
