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

# ============================================================
# Pre-flight checklist — checked BEFORE step 01 runs, so a missing
# file/folder is caught immediately instead of partway through a run
# (e.g. the Hindi run that got as far as step 05 with no data at all).
# ============================================================
echo ""
echo "============================================================"
echo " PRE-FLIGHT CHECKLIST for $STR_LANGUAGE"
echo "============================================================"
ALL_OK=true

check_item() {
    local ok="$1" label="$2" detail="$3"
    if [ "$ok" = true ]; then
        echo "  [OK]      $label"
    else
        echo "  [MISSING] $label ($detail)"
        ALL_OK=false
    fi
}

check_item "$([ -f "$VOCAB_PATH" ] && echo true || echo false)" \
    "vocab/${STR_LANGUAGE}_vocab.json" "$VOCAB_PATH"

check_item "$([ -f "$CLIP_CKPT" ] && echo true || echo false)" \
    "CLIP checkpoint" "$CLIP_CKPT"

# Data: either already-combined TSVs, or raw *_images.txt/*_labels.txt
# that 00c can combine — either counts as "have data" at this stage.
DATA_OK=true
for split in train val test; do
    tsv="$DATA_DIR/$split/$split.tsv"
    raw_images="$DATA_DIR/$split/${split}_images.txt"
    raw_labels="$DATA_DIR/$split/${split}_labels.txt"
    if [ -f "$tsv" ]; then
        continue
    elif [ -f "$raw_images" ] && [ -f "$raw_labels" ]; then
        continue
    else
        DATA_OK=false
        echo "  [MISSING] $split data (neither $tsv nor $raw_images + $raw_labels found)"
    fi
done
[ "$DATA_OK" = true ] && echo "  [OK]      train/val/test data (TSVs or raw images.txt+labels.txt)"
[ "$DATA_OK" = false ] && ALL_OK=false

# Images themselves: at least one file under each split's images/ folder.
IMAGES_OK=true
for split in train val test; do
    img_dir="$DATA_DIR/$split/images"
    if [ ! -d "$img_dir" ] || [ -z "$(find "$img_dir" -maxdepth 1 -type f 2>/dev/null)" ]; then
        IMAGES_OK=false
        echo "  [MISSING] $split/images/ (empty or missing: $img_dir)"
    fi
done
[ "$IMAGES_OK" = true ] && echo "  [OK]      train/val/test images/ folders (non-empty)"
[ "$IMAGES_OK" = false ] && ALL_OK=false

# Fonts: optional, only needed for synthetic data generation (steps 03/04).
FONTS_DIR_CHECK="$SCRIPT_DIR/../fonts/$STR_LANGUAGE"
if [ -d "$FONTS_DIR_CHECK" ] && [ -n "$(find "$FONTS_DIR_CHECK" -iname '*.ttf' -o -iname '*.otf' -o -iname '*.ttc' 2>/dev/null)" ]; then
    echo "  [OK]      fonts/$STR_LANGUAGE/ (synthetic data generation will run)"
else
    echo "  [OPTIONAL, not found] fonts/$STR_LANGUAGE/ — synthetic data generation will be SKIPPED,"
    echo "            training will use real data only. Not a blocker, but strongly recommended."
    echo "            >>> To fetch every available font for this language automatically:"
    echo "            >>>   ./setup_fonts_from_google.sh $STR_LANGUAGE"
fi

echo "============================================================"
if [ "$ALL_OK" = false ]; then
    echo ""
    echo "!! One or more REQUIRED items above are MISSING. Fix these first — continuing now"
    echo "!! would just fail partway through (as happened before). See README.md's"
    echo "!! 'Multi-language project layout' section for the expected file layout."
    exit 1
fi
echo ""
read -p ">>> All required items present. Proceed with the $STR_LANGUAGE pipeline? [y/N]: " preflight_confirm
if [ "$preflight_confirm" != "y" ] && [ "$preflight_confirm" != "Y" ]; then
    echo "Aborted — nothing was run."
    exit 0
fi

echo ""
echo "--- Ensuring vocab has common symbols (digits, punctuation, currency, Latin letters) ---"
"$PYTHON" "$SCRIPT_DIR/../add_common_symbols_to_vocab.py" --languages "$STR_LANGUAGE" --fix

CKPT_DIR="$SCRIPT_DIR/../checkpoints/$STR_LANGUAGE"
if [ -f "$CKPT_DIR/best.pt" ] || [ -f "$CKPT_DIR/resume_state.pt" ]; then
    echo ""
    echo "!! Existing checkpoint(s) found for $STR_LANGUAGE in $CKPT_DIR/"
    echo "!! If you've changed fonts, vocab, or want a genuinely fresh retrain (not a resume),"
    echo "!! these need clearing first — otherwise training may resume from stale state."
    read -p ">>> Clear existing checkpoints for $STR_LANGUAGE now? [y/N]: " clear_ckpt
    if [ "$clear_ckpt" = "y" ] || [ "$clear_ckpt" = "Y" ]; then
        rm -f "$CKPT_DIR"/resume_state.pt "$CKPT_DIR"/best.pt "$CKPT_DIR"/last.pt "$CKPT_DIR"/checkpoint_epoch_*.pt
        echo "Cleared. This run will start fresh from epoch 1 (not resume)."
    else
        echo "Keeping existing checkpoints — step 07 (train), if you proceed to it, will prompt"
        echo "again there about resuming vs. starting fresh."
    fi
fi

echo ""
echo "--- 01: environment setup (safe to re-run, skips if already correct) ---"
"$SCRIPT_DIR/01_setup.sh" "$STR_LANGUAGE"

echo ""
echo "--- 02: GPU check ---"
"$SCRIPT_DIR/02_check_gpu.sh" "$STR_LANGUAGE"

echo ""
echo "--- 00c: combine images.txt + labels.txt into TSVs (skips gracefully if not applicable) ---"
"$SCRIPT_DIR/00c_combine_images_labels.sh" "$STR_LANGUAGE"

MISSING_TSVS=()
for split_tsv in "$TRAIN_ANN" "$VAL_ANN" "$TEST_ANN"; do
    [ -f "$split_tsv" ] || MISSING_TSVS+=("$split_tsv")
done
if [ "${#MISSING_TSVS[@]}" -gt 0 ]; then
    echo ""
    echo "############################################################"
    echo "# STOPPING: no usable data for '$STR_LANGUAGE' yet."
    echo "# Missing:"
    for m in "${MISSING_TSVS[@]}"; do echo "#   $m"; done
    echo "#"
    echo "# 00c above either found no matching *_images.txt/*_labels.txt pair, or"
    echo "# you have a different raw format that needs a different combining step"
    echo "# (00b_split_annotations.sh for a single gt.txt, or write these TSVs by"
    echo "# hand). Continuing past this point would just fail later with a less"
    echo "# clear error (this is exactly what happened with an earlier Hindi run"
    echo "# that had no data in place yet)."
    echo "############################################################"
    exit 1
fi

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

    echo ""
    echo "--- 04b: confusion-focused synthetic data boost (round 2+; auto-skips if no prior results) ---"
    "$SCRIPT_DIR/04b_generate_confusion_boost.sh" "$STR_LANGUAGE"

    echo ""
    echo "--- 04c: minimal-pair confusion synthetic boost (round 2+; auto-skips if no prior results) ---"
    "$SCRIPT_DIR/04c_generate_confusion_pair_boost.sh" "$STR_LANGUAGE"
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
    if [ "${AUTO_EXTEND_EPOCHS:-1}" = "0" ]; then
        echo "--- 07: train (single fixed-epoch run; AUTO_EXTEND_EPOCHS=0 set) ---"
        # Passed explicitly (not just via the already-exported STR_LANGUAGE) so
        # this stays correct even if 07_train.sh is later refactored to not
        # trust an inherited env var — one argument, threaded through the whole
        # pipeline in a single script execution, is what actually prevents the
        # language drifting between steps (the failure mode that motivated this).
        "$SCRIPT_DIR/07_train.sh" "$STR_LANGUAGE"
        TRAIN_STATUS=$?
    else
        echo "--- 07: train (auto-extending: keeps raising the epoch ceiling and resuming"
        echo "    if training hits it WITHOUT early stopping genuinely triggering, instead"
        echo "    of a single fixed-epoch run — set AUTO_EXTEND_EPOCHS=0 to disable this and"
        echo "    do a single fixed-epoch run like before) ---"
        INCREMENT="${AUTO_EXTEND_INCREMENT:-100}"
        MAX_ROUNDS="${AUTO_EXTEND_MAX_ROUNDS:-3}"
        ABSOLUTE_MAX="${AUTO_EXTEND_ABSOLUTE_MAX:-500}"
        SUMMARY_FILE="$SCRIPT_DIR/../checkpoints/$STR_LANGUAGE/training_summary.txt"
        CURRENT_EPOCHS="$EPOCHS"
        ROUND=0
        TRAIN_STATUS=0

        while true; do
            ROUND=$((ROUND + 1))
            echo ""
            echo "    === Round $ROUND: training to $CURRENT_EPOCHS epochs ==="
            STR_EPOCHS="$CURRENT_EPOCHS" "$SCRIPT_DIR/07_train.sh" "$STR_LANGUAGE"
            TRAIN_STATUS=$?
            if [ $TRAIN_STATUS -ne 0 ]; then
                echo "!! Training itself failed (exit $TRAIN_STATUS) -- stopping, not an extension case."
                break
            fi

            if [ ! -f "$SUMMARY_FILE" ]; then
                echo "!! Could not find $SUMMARY_FILE after training -- can't check convergence status."
                break
            fi
            STOPPED_EARLY=$(grep -oE "Stopped early *: *(True|False)" "$SUMMARY_FILE" | grep -oE "True|False")
            echo "    $(grep "Best val_CER" "$SUMMARY_FILE")"
            echo "    Stopped early: ${STOPPED_EARLY:-unknown}"

            if [ "$STOPPED_EARLY" = "True" ]; then
                echo "    >>> Genuine convergence reached -- no further extension needed."
                break
            fi
            if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
                echo "!! Hit the epoch ceiling $ROUND time(s) without early stopping ever triggering --"
                echo "!! stopping extensions here (safety limit: AUTO_EXTEND_MAX_ROUNDS=$MAX_ROUNDS)."
                break
            fi
            NEXT_EPOCHS=$((CURRENT_EPOCHS + INCREMENT))
            if [ "$NEXT_EPOCHS" -gt "$ABSOLUTE_MAX" ]; then
                echo "!! Next extension ($NEXT_EPOCHS epochs) would exceed the absolute safety cap"
                echo "!! (AUTO_EXTEND_ABSOLUTE_MAX=$ABSOLUTE_MAX) -- stopping extensions here."
                break
            fi
            echo "    Did not converge -- extending to $NEXT_EPOCHS epochs and resuming..."
            CURRENT_EPOCHS="$NEXT_EPOCHS"
            export TRAIN_RESUME=1
        done
    fi

    if [ "${SKIP_EVAL:-0}" = "1" ]; then
        echo ""
        echo "SKIP_EVAL=1 set — skipping automatic evaluation. Run it later with:"
        echo "    ./08_evaluate.sh $STR_LANGUAGE"
        echo ""
        echo "############################################################"
        echo "# Full pipeline for $STR_LANGUAGE complete: data prep -> train. (evaluation skipped)"
        echo "############################################################"
    else
        echo ""
        echo "--- 08: evaluate ---"
        "$SCRIPT_DIR/08_evaluate.sh" "$STR_LANGUAGE"
        echo ""
        echo "############################################################"
        echo "# Full pipeline for $STR_LANGUAGE complete: data prep -> train -> evaluate."
        echo "############################################################"
    fi
else
    echo ""
    echo "Not starting training now. When you're ready, run (with the SAME language"
    echo "spelled out explicitly, not relying on an env var that might be stale):"
    echo "    ./07_train.sh $STR_LANGUAGE"
fi
