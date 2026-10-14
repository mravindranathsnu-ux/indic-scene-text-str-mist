#!/bin/bash
# config.sh — shared configuration, sourced by every step script in this
# folder. Edit the values below for your language/setup, then run the
# numbered step scripts in order (00 -> 09). Nothing in here should need
# editing per-run — only when you switch language, paths, or want to
# change a hyperparameter default.

set -euo pipefail

# ============================================================
# Language & project layout
# ============================================================
LANGUAGE="${STR_LANGUAGE:-telugu}"   # e.g. telugu, hindi, tamil, malayalam, kannada, bengali,
                                      # gujarati, punjabi, odia, assamese, marathi, meitei, urdu, english
                                      # Override without editing this file: STR_LANGUAGE=hindi ./07_train.sh
                                      # (NOT plain LANGUAGE= — that's a reserved Linux locale env var,
                                      #  e.g. LANGUAGE=en_IN:en is commonly already set by your shell,
                                      #  and would silently collide with a plain LANGUAGE override here)

# Resolves to the multilingual-str-clip project root regardless of where
# this script is invoked from (scripts/config.sh's own directory's parent).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="$PROJECT_ROOT/venv/bin/python"   # created by 01_setup.sh; falls back to system python3 if missing
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# PyTorch build to install in 01_setup.sh, matching your NVIDIA driver's
# max supported CUDA version (check with `nvidia-smi`, top-right corner —
# "CUDA Version: X.Y"). Leave empty to let pip pick its default (newest)
# build, which can be NEWER than your driver supports and cause
# "CUDA available: False" / "driver too old" in 02_check_gpu.sh.
# Pre-set to cu128 here (confirmed correct for driver 570.211.01 / CUDA
# 12.8) — change or blank this out if your driver/GPU differs.
TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"

# ============================================================
# Per-language data layout — data/<language>/{train,val,test}/{images/,*.tsv}
# ============================================================
DATA_DIR="$PROJECT_ROOT/data/$LANGUAGE"

TRAIN_ANN="$DATA_DIR/train/train.tsv"
TRAIN_IMAGE_ROOT="$DATA_DIR/train/images"

VAL_ANN="$DATA_DIR/val/val.tsv"
VAL_IMAGE_ROOT="$DATA_DIR/val/images"

TEST_ANN="$DATA_DIR/test/test.tsv"
TEST_IMAGE_ROOT="$DATA_DIR/test/images"

# Only relevant if you have ONE combined gt.txt instead of pre-split data —
# see 00b_split_annotations.sh.
RAW_GT_TXT="$DATA_DIR/gt.txt"

# ============================================================
# Vocab / fonts / model checkpoint locations
# ============================================================
CHAR_LISTS_DIR="$PROJECT_ROOT/char_lists"      # only used by 00a_convert_vocab.sh
VOCAB_DIR="$PROJECT_ROOT/vocab"
VOCAB_PATH="$VOCAB_DIR/${LANGUAGE}_vocab.json"

FONTS_DIR="$PROJECT_ROOT/fonts/$LANGUAGE"
FONT_CHECK_OUT_DIR="$PROJECT_ROOT/font_check_samples/$LANGUAGE"
SAMPLES_DIR="$TEST_IMAGE_ROOT"   # default --image_dir for 09_inference.sh when no argument is given.
                                  # Defaults to your real test images (not the empty samples/<language>/
                                  # placeholder folder) so ./09_inference.sh works with zero setup. To
                                  # test on genuinely new/unseen images instead, either pass a path
                                  # directly (./09_inference.sh /path/to/images) or point this at
                                  # samples/$LANGUAGE after adding your own images there.

CLIP_CKPT="$PROJECT_ROOT/weights/ViT-B-16.pt"    # switched from ViT-B-32: finer patches (16x16 vs
                                                  # 32x32 px) for better fine-grained glyph discrimination
                                                  # -- put your ViT-B-16.pt file in weights/

OUT_DIR="$PROJECT_ROOT/checkpoints/$LANGUAGE"
RESULTS_DIR="$PROJECT_ROOT/results"

# ============================================================
# Synthetic data generation
# ============================================================
SYNTHETIC_OUT_DIR="$DATA_DIR/train/synthetic"
TRAIN_AUGMENTED_ANN="$DATA_DIR/train/train_augmented.tsv"
SAMPLES_PER_WORD=6
LENGTH_BOOST="${STR_LENGTH_BOOST:-0.8}"   # override without editing: STR_LENGTH_BOOST=1.2 ./04_generate_synthetic_data.sh
                                            # (auto-computed by run_pretrain_pipeline.sh via analyze_length_distribution.py
                                            # unless SKIP_LENGTH_ANALYSIS=1)
CORPUS_FILE=""   # optional: path to an extra word-list .txt for vocabulary expansion; leave empty to skip

# ============================================================
# Training hyperparameters
# ============================================================
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
EPOCHS=100
PATIENCE=15
WARMUP_FROZEN_EPOCHS=3
UNFREEZE_LAST_N=6   # raised from 4 -- more fine-tuned backbone capacity, justified now that
                    # we have real data volume (34k+ train images) to support it without overfitting
LR=3e-4
BACKBONE_LR=3e-5

# ============================================================
# Decoding (evaluate.py / inference.py)
# ============================================================
BEAM_SIZE=5
LENGTH_PENALTY=1.0

# ============================================================
# Sanity check
# ============================================================
OVERFIT_STEPS=300
OVERFIT_LR=1e-3

echo "[config] language=$LANGUAGE  project_root=$PROJECT_ROOT  python=$PYTHON"
