#!/bin/bash
# report_data_counts.sh — prints real (train/val/test) and synthetic sample
# counts for every language that has data in place, in one summary table.
# Auto-discovers languages from data/ — no need to list them manually, and
# it stays current as you add more.
#
# Usage:
#   ./report_data_counts.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

if [ ! -d "$DATA_DIR" ]; then
    echo "No data/ directory found at $DATA_DIR"
    exit 1
fi

count_lines() {
    # Prints a line count, or "-" if the file doesn't exist.
    if [ -f "$1" ]; then
        wc -l < "$1" | tr -d ' '
    else
        echo "-"
    fi
}

printf "%-14s %10s %10s %10s %14s %16s\n" "Language" "Train" "Val" "Test" "Synthetic" "Train+Synthetic"
printf "%-14s %10s %10s %10s %14s %16s\n" "--------" "-----" "---" "----" "---------" "----------------"

for lang_dir in "$DATA_DIR"/*/; do
    lang="$(basename "$lang_dir")"
    train_n=$(count_lines "$DATA_DIR/$lang/train/train.tsv")
    val_n=$(count_lines "$DATA_DIR/$lang/val/val.tsv")
    test_n=$(count_lines "$DATA_DIR/$lang/test/test.tsv")
    synth_n=$(count_lines "$DATA_DIR/$lang/train/synthetic/synthetic.tsv")
    aug_n=$(count_lines "$DATA_DIR/$lang/train/train_augmented.tsv")

    # Skip languages with no real data at all yet (still-empty placeholder dirs).
    if [ "$train_n" = "-" ] && [ "$val_n" = "-" ] && [ "$test_n" = "-" ]; then
        continue
    fi

    printf "%-14s %10s %10s %10s %14s %16s\n" "$lang" "$train_n" "$val_n" "$test_n" "$synth_n" "$aug_n"
done

echo ""
echo "Train/Val/Test: real samples only (from train.tsv/val.tsv/test.tsv)."
echo "Synthetic: generated synthetic samples (from train/synthetic/synthetic.tsv), '-' if not yet generated."
echo "Train+Synthetic: what training actually used, if train_augmented.tsv exists (real + synthetic combined)."
