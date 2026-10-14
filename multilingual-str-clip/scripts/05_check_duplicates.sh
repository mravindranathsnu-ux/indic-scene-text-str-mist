#!/bin/bash
# NEW: pass the language as an explicit argument, e.g. ./05_check_duplicates.sh malayalam
# — this always wins over any stale STR_LANGUAGE left exported in your shell
# from an earlier command (a real bug this project hit), so prefer it over
# relying on STR_LANGUAGE alone.
# Step 5: Check for byte-identical images appearing under different
# filenames in both train and test — a real-world leakage vector that
# path-string-based splitting can't catch. Run this after step 04 (uses
# train_augmented.tsv if it exists, otherwise the plain train.tsv).
#
# Output is also saved to results/<language>_duplicate_check.txt (in
# addition to the normal terminal output) -- share that file directly
# instead of pasting terminal output, if it's ever needed for reference.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"
mkdir -p results
LOG_FILE="results/${STR_LANGUAGE}_duplicate_check.txt"

{
    echo "=== Step 5: Check for duplicate images between train and test ==="
    echo "Language:  $STR_LANGUAGE"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    ANN_FOR_CHECK="$TRAIN_ANN"
    if [ -f "$TRAIN_AUGMENTED_ANN" ]; then
        ANN_FOR_CHECK="$TRAIN_AUGMENTED_ANN"
    fi

    "$PYTHON" check_image_duplicates.py \
        --train_image_root "$TRAIN_IMAGE_ROOT" \
        --test_image_root "$TEST_IMAGE_ROOT" \
        --train_ann "$ANN_FOR_CHECK" \
        --test_ann "$TEST_ANN"
    PYTHON_STATUS=$?

    echo ""
    echo ">>> If any byte-identical images were reported above, investigate before"
    echo ">>> training — that image's label would otherwise leak into training."

    # The block's own exit status is normally whatever its LAST command
    # returns -- which would be the echo above (always 0), silently masking
    # a real failure from check_image_duplicates.py. Force it to reflect
    # the actual check's exit code instead (caught by testing this exact
    # failure case before shipping).
    (exit "$PYTHON_STATUS")
} 2>&1 | tee "$LOG_FILE"
CHECK_STATUS=${PIPESTATUS[0]}

echo ""
echo ">>> Log saved: $LOG_FILE"
exit "$CHECK_STATUS"
