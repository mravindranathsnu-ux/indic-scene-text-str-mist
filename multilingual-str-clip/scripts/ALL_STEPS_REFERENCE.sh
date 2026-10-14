#!/bin/bash
# ============================================================================
# ALL_STEPS_REFERENCE.sh — DOCUMENTATION/REFERENCE ONLY.
#
# Every numbered script accepts an explicit language as its first argument
# (e.g. ./07_train.sh malayalam), which always wins over any stale
# STR_LANGUAGE left exported in your shell. Exception: 09_inference.sh's
# first argument is the image directory (use
# STR_LANGUAGE=<language> ./09_inference.sh /path/to/images instead).
#
# RECOMMENDED USAGE: run_<language>.sh (e.g. ./run_malayalam.sh) runs the
# ENTIRE pipeline for one language in a single execution — setup through
# the sanity check, then asks whether to continue straight into training.
# The language is threaded through explicitly at every step within that
# one execution, so it cannot drift or fall back to a stale/default value
# partway through — this is the safest way to run a language end-to-end.
# ============================================================================


# ============================================================================
# ---- config.sh ----
# ============================================================================
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
LANGUAGE="${1:-${STR_LANGUAGE:-telugu}}"   # priority: 1) explicit positional arg, e.g. ./07_train.sh malayalam
                                            #           2) STR_LANGUAGE env var, e.g. STR_LANGUAGE=hindi ./07_train.sh
                                            #           3) default: telugu
                                            # PREFER THE POSITIONAL ARG FORM — an env var set with plain
                                            # `export STR_LANGUAGE=...` lingers for the rest of your shell
                                            # session and can silently apply to a LATER, unrelated command
                                            # for a different language (this has actually happened in this
                                            # project). A positional arg is typed fresh every time and always
                                            # wins over any stale env var, so it can't have this problem.
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

if [ -n "${1:-}" ]; then
    LANGUAGE_SOURCE="positional arg"
elif [ -n "${STR_LANGUAGE:-}" ]; then
    LANGUAGE_SOURCE="STR_LANGUAGE env var (verify this isn't stale from an earlier command!)"
else
    LANGUAGE_SOURCE="default"
fi
echo "[config] language=$LANGUAGE (source: $LANGUAGE_SOURCE)  project_root=$PROJECT_ROOT  python=$PYTHON"


# ============================================================================
# ---- 00a_convert_vocab.sh ----
# ============================================================================
#!/bin/bash
# Step 0a (OPTIONAL): Convert plain character-list .txt files (one char
# per line — the common CRNN/CTC vocab format) into the JSON vocab files
# vocab.py expects. Skip this script entirely if vocab/<language>_vocab.json
# already exists.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 0a: Convert char-list files -> vocab JSON ==="
cd "$PROJECT_ROOT"

if [ ! -d "$CHAR_LISTS_DIR" ]; then
    echo "No $CHAR_LISTS_DIR directory found — nothing to convert."
    echo "(This step is only needed if you have raw .txt char-list files to convert;"
    echo " if vocab/${LANGUAGE}_vocab.json already exists, skip straight to step 01.)"
    exit 0
fi

"$PYTHON" convert_char_list.py --input_dir "$CHAR_LISTS_DIR" --output_dir "$VOCAB_DIR"
echo "Vocab files written to $VOCAB_DIR/"


# ============================================================================
# ---- 00b_split_annotations.sh ----
# ============================================================================
#!/bin/bash
# Step 0b (OPTIONAL / ALTERNATE): Split a single combined gt.txt into
# train/val/test TSVs. SKIP this script if your image database already
# ships train/val/test as separate, pre-split image sets (the expected
# common case — see the "Multi-language project layout" in README.md).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 0b: Split $RAW_GT_TXT into train/val/test ==="
cd "$PROJECT_ROOT"

if [ ! -f "$RAW_GT_TXT" ]; then
    echo "No $RAW_GT_TXT found — nothing to split."
    echo "(Skip this step if you already have separate pre-split train/val/test data.)"
    exit 0
fi

"$PYTHON" split_annotations.py \
    --gt "$RAW_GT_TXT" \
    --out_dir "$DATA_DIR" \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --dedup

echo "NOTE: split_annotations.py writes train.tsv/val.tsv/test.tsv directly into"
echo "      $DATA_DIR — move each into its own train/, val/, test/ subfolder"
echo "      (alongside that split's own images/) to match this project's layout,"
echo "      e.g.: mkdir -p $DATA_DIR/train && mv $DATA_DIR/train.tsv $TRAIN_ANN"


# ============================================================================
# ---- 00c_combine_images_labels.sh ----
# ============================================================================
#!/bin/bash
# Step 0c (OPTIONAL): Combine separate train_images.txt/train_labels.txt
# (and val_/test_ equivalents) into this project's train.tsv/val.tsv/
# test.tsv format. Use this if your raw data ships as parallel image-list
# and label-list files rather than already-combined TSVs.
#
# Expects, for each split, these files already in place:
#   $DATA_DIR/<split>/<split>_images.txt   (one image filename/path per line)
#   $DATA_DIR/<split>/<split>_labels.txt   (one label per line, same order)
#
# Writes:
#   $DATA_DIR/<split>/<split>.tsv
#
# IMPORTANT — check your raw *_images.txt content before running, since
# the right prefix handling depends on what's already in there:
#   - Bare filenames (e.g. "img_0001.jpg")             -> set IMAGE_PREFIX="images/", STRIP_PREFIX=""
#   - Already full paths (e.g. "images/img_0001.jpg")  -> set IMAGE_PREFIX="",        STRIP_PREFIX=""
#   - Raw dataset's OWN internal structure baked in
#     (e.g. "train/images/img_0001.jpg", which would
#     duplicate part of --train_image_root if prefixed
#     further)                                          -> set STRIP_PREFIX="$split/images/" (default below)
# Default here assumes the third case (STRIP_PREFIX="$split/images/" per
# split, IMAGE_PREFIX="") since that's what this project has actually
# run into — check a few lines of your own *_images.txt against
# `ls $DATA_DIR/<split>/images/` first and adjust below if different.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 0c: Combine images.txt + labels.txt -> TSV for $LANGUAGE ==="
cd "$PROJECT_ROOT"

IMAGE_PREFIX=""   # prepended AFTER stripping — see comment block above

for split in train val test; do
    split_dir="$DATA_DIR/$split"
    images_txt="$split_dir/${split}_images.txt"
    labels_txt="$split_dir/${split}_labels.txt"
    out_tsv="$split_dir/${split}.tsv"
    strip_prefix="$split/images/"   # matches this project's actual raw-data case; adjust per your data

    if [ ! -f "$images_txt" ] || [ ! -f "$labels_txt" ]; then
        echo "  [$split] skipped — $images_txt or $labels_txt not found"
        continue
    fi

    "$PYTHON" combine_images_labels.py \
        --images_txt "$images_txt" \
        --labels_txt "$labels_txt" \
        --output "$out_tsv" \
        --image_prefix "$IMAGE_PREFIX" \
        --strip_prefix "$strip_prefix"
done

echo ""
echo "Done. Any split reported 'skipped' above had no matching *_images.txt/"
echo "*_labels.txt pair under $DATA_DIR/<split>/ — check filenames if unexpected."
echo ""
echo ">>> ACTION REQUIRED: spot-check the output TSVs before running sanity_check.py —"
echo ">>> e.g. 'head -3 $DATA_DIR/train/train.tsv' and confirm each image_path, when"
echo ">>> joined with --train_image_root, actually points at a real file:"
echo ">>>   ls \"\$(dirname \"$DATA_DIR/train/images/<first image_path from the TSV>\")\""


# ============================================================================
# ---- 01_setup.sh ----
# ============================================================================
#!/bin/bash
# Step 1: Create a virtualenv and install dependencies. Run this once
# per machine (not per language).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 1: Environment setup ==="
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# Install torch/torchvision from a SPECIFIC CUDA build, matching your
# driver — pip's default (unpinned) install grabs the newest CUDA build
# available, which can be newer than your driver supports (a driver can
# run OLDER CUDA builds fine, but not newer ones). Run 02_check_gpu.sh
# once to find your driver's max CUDA version (via `nvidia-smi`), set
# TORCH_CUDA_INDEX_URL in config.sh to match.
if [ -n "$TORCH_CUDA_INDEX_URL" ]; then
    DESIRED_CUDA_TAG=$(echo "$TORCH_CUDA_INDEX_URL" | grep -oE 'cu[0-9]+' || true)
    CURRENT_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
    CURRENT_CUDA_TAG="cu$(echo "$CURRENT_CUDA_VERSION" | tr -d '.')"

    if [ -n "$DESIRED_CUDA_TAG" ] && [ "$CURRENT_CUDA_TAG" = "$DESIRED_CUDA_TAG" ]; then
        echo "torch already installed with matching CUDA build ($CURRENT_CUDA_TAG) — skipping reinstall."
    else
        echo "Installing torch/torchvision from $TORCH_CUDA_INDEX_URL (currently: ${CURRENT_CUDA_TAG:-none installed})..."
        # --force-reinstall is REQUIRED here: if some torch build is already
        # installed (any version), plain `pip install torch` sees
        # "requirement already satisfied" and SKIPS it entirely even if
        # it's the wrong CUDA build. We only reach this branch when we've
        # already confirmed a mismatch above, so the (slow, large-download)
        # reinstall only happens when actually needed, not on every run.
        pip install --force-reinstall torch torchvision --index-url "$TORCH_CUDA_INDEX_URL"
    fi
else
    echo "TORCH_CUDA_INDEX_URL not set in config.sh — using pip's default torch build."
    echo "If 02_check_gpu.sh then reports a driver/CUDA mismatch, set TORCH_CUDA_INDEX_URL"
    echo "in config.sh (e.g. https://download.pytorch.org/whl/cu128) and re-run this script."
fi

pip install -r requirements.txt

echo "Setup complete."
echo "Every later step script calls venv's python directly (no need to"
echo "manually 'source venv/bin/activate' again), but you can if you prefer"
echo "to run project commands by hand instead of via these scripts."


# ============================================================================
# ---- 02_check_gpu.sh ----
# ============================================================================
#!/bin/bash
# Step 2: Confirm GPU is visible and actually usable (not just "available"
# per the driver). Run once per machine; re-run after any driver/CUDA/
# PyTorch reinstall.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 2: GPU / environment check ==="
cd "$PROJECT_ROOT"
"$PYTHON" check_gpu.py


# ============================================================================
# ---- 03_verify_fonts.sh ----
# ============================================================================
#!/bin/bash
# Step 3 (OPTIONAL — only needed if generating synthetic data in step 04):
# Render a stress-test word with every font in fonts/<language>/ and
# report which ones fail to load. STILL REQUIRES a human to open the
# output PNGs and visually confirm correct shaping (conjuncts/matras/
# ligatures) before trusting a font — this script can't verify that part
# automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 3: Verify fonts for $LANGUAGE ==="
cd "$PROJECT_ROOT"

if [ ! -d "$FONTS_DIR" ]; then
    echo "No $FONTS_DIR directory found — skipping (only needed for synthetic data generation)."
    exit 0
fi

"$PYTHON" verify_fonts.py --fonts_dir "$FONTS_DIR" --language "$LANGUAGE" --out_dir "$FONT_CHECK_OUT_DIR"

echo ""
echo ">>> ACTION REQUIRED: open the PNGs in $FONT_CHECK_OUT_DIR/ and visually"
echo ">>> confirm the script rendered correctly before running step 04."


# ============================================================================
# ---- 04_generate_synthetic_data.sh ----
# ============================================================================
#!/bin/bash
# Step 4 (OPTIONAL but recommended once real-data hyperparameter tuning
# plateaus): Render synthetic word images to supplement real training
# data, then combine into train_augmented.tsv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 4: Generate synthetic training data for $LANGUAGE ==="
cd "$PROJECT_ROOT"

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
echo ">>> ACTION REQUIRED: open a handful of images in $SYNTHETIC_OUT_DIR/images/"
echo ">>> and visually confirm the script rendered correctly (garbled synthetic"
echo ">>> labels would actively hurt training)."


# ============================================================================
# ---- 05_check_duplicates.sh ----
# ============================================================================
#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./05_check_duplicates.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 5: Check for byte-identical images appearing under different
# filenames in both train and test — a real-world leakage vector that
# path-string-based splitting can't catch. Run this after step 04 (uses
# train_augmented.tsv if it exists, otherwise the plain train.tsv).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 5: Check for duplicate images between train and test ==="
cd "$PROJECT_ROOT"

ANN_FOR_CHECK="$TRAIN_ANN"
if [ -f "$TRAIN_AUGMENTED_ANN" ]; then
    ANN_FOR_CHECK="$TRAIN_AUGMENTED_ANN"
fi

"$PYTHON" check_image_duplicates.py \
    --train_image_root "$TRAIN_IMAGE_ROOT" \
    --test_image_root "$TEST_IMAGE_ROOT" \
    --train_ann "$ANN_FOR_CHECK" \
    --test_ann "$TEST_ANN"

echo ""
echo ">>> If any byte-identical images were reported above, investigate before"
echo ">>> training — that image's label would otherwise leak into training."


# ============================================================================
# ---- 06_sanity_check.sh ----
# ============================================================================
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


# ============================================================================
# ---- 07_train.sh ----
# ============================================================================
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

YES_FLAG=()
if [ "${TRAIN_YES:-0}" = "1" ]; then
    YES_FLAG=(--yes)
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
    "${YES_FLAG[@]}"

echo ""
echo "Training complete. Checkpoints: $OUT_DIR/{best.pt,last.pt}"


# ============================================================================
# ---- 08_evaluate.sh ----
# ============================================================================
#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./08_evaluate.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 8: Evaluate the trained checkpoint on the held-out test set —
# reports WRR/CRR (and CER/WER), and writes per-sample predictions to CSV.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 8: Evaluate $LANGUAGE model ==="
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"

"$PYTHON" evaluate.py \
    --ckpt "$OUT_DIR/best.pt" \
    --clip_ckpt "$CLIP_CKPT" \
    --test_ann "$TEST_ANN" \
    --test_image_root "$TEST_IMAGE_ROOT" \
    --decode beam --beam_size "$BEAM_SIZE" --length_penalty "$LENGTH_PENALTY" \
    --out_csv "$RESULTS_DIR/${LANGUAGE}_test_predictions.csv"

echo ""
echo "Per-sample predictions: $RESULTS_DIR/${LANGUAGE}_test_predictions.csv"


# ============================================================================
# ---- 09_inference.sh ----
# ============================================================================
#!/bin/bash
# Step 9: Run inference on new images with the trained checkpoint.
# Usage: ./09_inference.sh /path/to/image_dir
#        (defaults to samples/<language>/ if no argument given)
# NOTE: this script's $1 is the image directory, not a language name (unlike
# the other numbered scripts) — set STR_LANGUAGE=<language> to pick which
# language's checkpoint/vocab to use, e.g.:
#     STR_LANGUAGE=malayalam ./09_inference.sh /path/to/images
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

"$PYTHON" inference.py \
    --ckpt "$OUT_DIR/best.pt" \
    --clip_ckpt "$CLIP_CKPT" \
    --image_dir "$IMAGE_DIR" \
    --decode beam --beam_size "$BEAM_SIZE" --length_penalty "$LENGTH_PENALTY" \
    --out_csv "$OUT_CSV"


# ============================================================================
# ---- run_pretrain_pipeline.sh ----
# ============================================================================
#!/bin/bash
# run_pretrain_pipeline.sh — runs every step for one language, in order:
# setup through the sanity check, pausing to confirm at the two steps that
# need a human to look at something (font shaping PNGs, sanity check
# verdict) — then asks whether to continue straight into training (07).
# The language is set ONCE from this script's argument and passed through
# explicitly to every sub-script call in this same execution, so it can't
# drift or fall back to a stale/default value partway through (a real bug
# this project hit when 07_train.sh was run as a separate, later command).
#
# Usage:
#   ./run_pretrain_pipeline.sh telugu
#   ./run_pretrain_pipeline.sh tamil
#
# This is the shared logic behind the per-language wrapper scripts
# (run_telugu.sh, run_tamil.sh, ...) — those just call this with the right
# language name, so there's one place to fix things instead of N copies.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <language>   (e.g. $0 telugu)"
    exit 1
fi
export STR_LANGUAGE="$1"
source "$SCRIPT_DIR/config.sh"

pause() {
    echo ""
    read -p ">>> Review the output above. Press Enter to continue, or Ctrl+C to stop here... " _
}

echo "############################################################"
echo "# Pre-training pipeline for: $STR_LANGUAGE"
echo "############################################################"

echo ""
echo "--- 01: environment setup (safe to re-run, skips if already correct) ---"
"$SCRIPT_DIR/01_setup.sh" "$STR_LANGUAGE"

echo ""
echo "--- 02: GPU check ---"
"$SCRIPT_DIR/02_check_gpu.sh" "$STR_LANGUAGE"

echo ""
echo "--- 00c: combine images.txt + labels.txt into TSVs (skips gracefully if not applicable) ---"
"$SCRIPT_DIR/00c_combine_images_labels.sh" "$STR_LANGUAGE"

FONTS_DIR="$SCRIPT_DIR/../fonts/$STR_LANGUAGE"
HAS_FONTS=false
if [ -d "$FONTS_DIR" ] && [ -n "$(find "$FONTS_DIR" -iname '*.ttf' -o -iname '*.otf' -o -iname '*.ttc' 2>/dev/null)" ]; then
    HAS_FONTS=true
fi

if [ "$HAS_FONTS" = true ]; then
    echo ""
    echo "--- 03: verify fonts ---"
    "$SCRIPT_DIR/03_verify_fonts.sh" "$STR_LANGUAGE"
    pause

    echo ""
    echo "--- length-distribution analysis (suggests --length_boost) ---"
    if [ "${SKIP_LENGTH_ANALYSIS:-0}" = "1" ]; then
        echo "SKIP_LENGTH_ANALYSIS=1 set — using LENGTH_BOOST from config.sh as-is."
    elif [ -f "$TRAIN_ANN" ]; then
        "$PYTHON" "$SCRIPT_DIR/../analyze_length_distribution.py" --train_ann "$TRAIN_ANN"
        SUGGESTED_BOOST=$("$PYTHON" "$SCRIPT_DIR/../analyze_length_distribution.py" \
            --train_ann "$TRAIN_ANN" --print_boost_only)
        echo "Using suggested --length_boost=$SUGGESTED_BOOST for step 04 below."
        echo "(This is a heuristic, not a guarantee — see the NOTE above. Override with"
        echo " SKIP_LENGTH_ANALYSIS=1 to use config.sh's LENGTH_BOOST unchanged instead.)"
        export STR_LENGTH_BOOST="$SUGGESTED_BOOST"
    else
        echo "No train.tsv found yet at $TRAIN_ANN — skipping, using config.sh's LENGTH_BOOST as-is."
    fi

    echo ""
    echo "--- 04: generate synthetic data + combine with real data ---"
    "$SCRIPT_DIR/04_generate_synthetic_data.sh" "$STR_LANGUAGE"
    echo ""
    echo ">>> Spot-check a few images in the synthetic output folder before continuing"
    echo ">>> (see the ACTION REQUIRED note printed by step 04 above for the exact path)."
    pause
else
    echo ""
    echo "--- 03/04: SKIPPED — no fonts found in $FONTS_DIR, so no synthetic data generation. ---"
    echo "    Training will use real data only. Add fonts there and re-run this script if you"
    echo "    want synthetic augmentation (recommended once real-data tuning plateaus)."
fi

echo ""
echo "--- 05: check for duplicate images between train and test ---"
"$SCRIPT_DIR/05_check_duplicates.sh" "$STR_LANGUAGE"

echo ""
echo "--- 06: sanity check (data, preprocessing, tiny-batch overfit tests) ---"
"$SCRIPT_DIR/06_sanity_check.sh" "$STR_LANGUAGE"

echo ""
echo "############################################################"
echo "# Pre-training pipeline for $STR_LANGUAGE complete."
echo "# Review the OVERALL VERDICT printed by step 06 above."
echo "############################################################"
echo ""
read -p ">>> Proceed straight to training $STR_LANGUAGE now? [y/N]: " proceed
if [ "$proceed" = "y" ] || [ "$proceed" = "Y" ]; then
    echo ""
    echo "--- 07: train ---"
    # Passed explicitly (not just via the already-exported STR_LANGUAGE) so
    # this stays correct even if 07_train.sh is later refactored to not
    # trust an inherited env var — one argument, threaded through the whole
    # pipeline in a single script execution, is what actually prevents the
    # language drifting between steps (the failure mode that motivated this).
    "$SCRIPT_DIR/07_train.sh" "$STR_LANGUAGE"
else
    echo ""
    echo "Not starting training now. When you're ready, run (with the SAME language"
    echo "spelled out explicitly, not relying on an env var that might be stale):"
    echo "    ./07_train.sh $STR_LANGUAGE"
fi

