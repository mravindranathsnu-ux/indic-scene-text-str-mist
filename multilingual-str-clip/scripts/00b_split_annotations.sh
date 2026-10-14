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
