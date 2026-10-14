#!/bin/bash
# auto_extend_training.sh — trains a language, and if it hits its epoch
# ceiling WITHOUT genuinely converging (early stopping via --patience
# never triggered, i.e. "Stopped early: False" in the summary), resumes
# automatically with a higher ceiling and continues -- repeating up to a
# safety limit -- instead of requiring a human to guess a fixed epoch
# count upfront and manually restart if it wasn't enough.
#
# The actual "when to stop" decision is still made by train.py's own
# --patience mechanism (already proven across every language in this
# project) -- this script's only job is to make sure the epoch CEILING
# never cuts that off prematurely, the way Telugu's round-2 100-epoch
# ceiling did (best epoch 95/100, val_CER still not clearly plateaued).
#
# Usage:
#   ./auto_extend_training.sh <language>
#
# Env var overrides:
#   AUTO_EXTEND_INCREMENT     epochs to add per extension (default 100)
#   AUTO_EXTEND_MAX_ROUNDS    max number of extensions before giving up (default 3)
#   AUTO_EXTEND_ABSOLUTE_MAX  hard ceiling regardless of rounds (default 500)
#
# IMPORTANT: this assumes 07_train.sh accepts an epoch-count override via
# STR_EPOCHS and a resume trigger via TRAIN_RESUME=1, matching this
# project's established STR_<PARAM> override convention elsewhere
# (STR_BACKBONE_LR, STR_UNFREEZE_LAST_N) and the TRAIN_RESUME=1 pattern
# confirmed earlier in this project. VERIFY both against your actual
# config.sh/07_train.sh before relying on this -- if either name differs,
# update the two lines marked VERIFY below to match.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

INCREMENT="${AUTO_EXTEND_INCREMENT:-100}"
MAX_ROUNDS="${AUTO_EXTEND_MAX_ROUNDS:-3}"
ABSOLUTE_MAX="${AUTO_EXTEND_ABSOLUTE_MAX:-500}"
SUMMARY_FILE="$PROJECT_ROOT/checkpoints/$STR_LANGUAGE/training_summary.txt"

CURRENT_EPOCHS="$EPOCHS"
ROUND=0

echo "############################################################"
echo "# Auto-extending training for $STR_LANGUAGE"
echo "# Starting ceiling: $CURRENT_EPOCHS epochs (extends by $INCREMENT if the cap is hit"
echo "# without genuine convergence, up to $MAX_ROUNDS extension(s), absolute max $ABSOLUTE_MAX)"
echo "############################################################"

while true; do
    ROUND=$((ROUND + 1))
    echo ""
    echo "=== Round $ROUND: training to $CURRENT_EPOCHS epochs ==="
    STR_EPOCHS="$CURRENT_EPOCHS" "$SCRIPT_DIR/07_train.sh" "$STR_LANGUAGE"   # VERIFY: STR_EPOCHS
    TRAIN_STATUS=$?
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "!! Training itself failed (exit $TRAIN_STATUS) -- stopping, not an extension case."
        exit 1
    fi

    if [ ! -f "$SUMMARY_FILE" ]; then
        echo "!! Could not find $SUMMARY_FILE after training -- can't check convergence status."
        echo "!! Check its actual output path/name and adjust SUMMARY_FILE in this script if needed."
        exit 1
    fi

    STOPPED_EARLY=$(grep -oE "Stopped early *: *(True|False)" "$SUMMARY_FILE" | grep -oE "True|False")
    BEST_LINE=$(grep "Best val_CER" "$SUMMARY_FILE")

    echo ""
    echo "--- Round $ROUND result ---"
    echo "$BEST_LINE"
    echo "Stopped early: ${STOPPED_EARLY:-unknown}"

    if [ "$STOPPED_EARLY" = "True" ]; then
        echo ""
        echo ">>> Genuine convergence reached (early stopping triggered on its own) -- no extension needed."
        echo ">>> Final result stands as-is."
        break
    fi

    if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
        echo ""
        echo "!! Hit the epoch ceiling $ROUND time(s) without early stopping ever triggering --"
        echo "!! stopping extensions here (safety limit: AUTO_EXTEND_MAX_ROUNDS=$MAX_ROUNDS)."
        echo "!! This run may genuinely still be improving slowly, or something other than raw"
        echo "!! epoch count may be limiting it (worth investigating rather than extending further)."
        break
    fi

    NEXT_EPOCHS=$((CURRENT_EPOCHS + INCREMENT))
    if [ "$NEXT_EPOCHS" -gt "$ABSOLUTE_MAX" ]; then
        echo ""
        echo "!! Next extension ($NEXT_EPOCHS epochs) would exceed the absolute safety cap"
        echo "!! (AUTO_EXTEND_ABSOLUTE_MAX=$ABSOLUTE_MAX) -- stopping extensions here."
        break
    fi

    echo ""
    echo "Did not converge (hit the ceiling without early stopping) -- extending to $NEXT_EPOCHS"
    echo "epochs and resuming from where this round left off..."
    CURRENT_EPOCHS="$NEXT_EPOCHS"
    export TRAIN_RESUME=1   # VERIFY: matches this project's established resume-trigger convention
done

echo ""
echo "############################################################"
echo "# Done after $ROUND round(s). Final summary: $SUMMARY_FILE"
echo "############################################################"
