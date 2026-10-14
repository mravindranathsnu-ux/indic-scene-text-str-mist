#!/bin/bash
# decode_sweep.sh — interactive Tier-1 decode-parameter sweep (beam size,
# length_penalty). Zero retraining cost -- just re-runs evaluate.py against
# an EXISTING checkpoint with different decoding settings, to check for
# free accuracy gains before touching training at all.
#
# Usage:
#   ./scripts/decode_sweep.sh <language>
#   ./scripts/decode_sweep.sh tamil
#
# Safe to run while another language is training -- this only reads the
# given language's checkpoint (read-only inference), no writes to any
# checkpoint. If the GPU is genuinely full from a concurrent training run,
# you'll see a CUDA out-of-memory error -- that's real GPU contention, not
# a bug in this script; wait for the other run to finish and retry.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CKPT="$PROJECT_ROOT/checkpoints/$LANGUAGE/best.pt"
CLIP_CKPT="$PROJECT_ROOT/weights/ViT-B-16.pt"
TEST_ANN="$PROJECT_ROOT/data/$LANGUAGE/test/test.tsv"
TEST_IMAGE_ROOT="$PROJECT_ROOT/data/$LANGUAGE/test/images"
OUT_DIR="$PROJECT_ROOT/results/decode_sweep_$LANGUAGE"
mkdir -p "$OUT_DIR"

if [ ! -f "$CKPT" ]; then
    echo "!! No checkpoint found at $CKPT -- nothing to sweep for $LANGUAGE."
    exit 1
fi

run_eval() {
    local name="$1" beam_size="$2" length_penalty="$3"
    local out_csv="$OUT_DIR/${LANGUAGE}_${name}.csv"
    echo ""
    echo "=== Running: $name (beam_size=$beam_size, length_penalty=$length_penalty) ==="
    "$PYTHON" evaluate.py \
        --ckpt "$CKPT" \
        --clip_ckpt "$CLIP_CKPT" \
        --test_ann "$TEST_ANN" \
        --test_image_root "$TEST_IMAGE_ROOT" \
        --decode beam \
        --beam_size "$beam_size" \
        --length_penalty "$length_penalty" \
        --out_csv "$out_csv" \
        --yes
    if [ $? -ne 0 ]; then
        echo "!! Run '$name' failed -- see output above."
        return 1
    fi
    echo ">>> Saved: $out_csv (+ matching _report.txt/.json alongside it)"
}

run_option() {
    case "$1" in
        1) run_eval "baseline_beam5_lp1.0"  5  1.0 ;;
        2) run_eval "beam10_lp1.0"          10 1.0 ;;
        3) run_eval "beam10_lp0.5"          10 0.5 ;;
        4) run_eval "beam10_lp0.7"          10 0.7 ;;
        5) run_eval "beam10_lp1.3"          10 1.3 ;;
        7) run_eval "beam5_lp1.5"           5  1.5 ;;
        8) run_eval "beam5_lp1.7"           5  1.7 ;;
        9) run_eval "beam5_lp2.0"           5  2.0 ;;
        *) echo "!! Unknown option: $1" ; return 1 ;;
    esac
}

echo "############################################################"
echo "# Decode-parameter sweep for $LANGUAGE"
echo "# Checkpoint: $CKPT"
echo "############################################################"
echo ""
echo "  1) Baseline        (beam_size=5,  length_penalty=1.0)  -- sanity check, should match your known result"
echo "  2) Larger beam     (beam_size=10, length_penalty=1.0)"
echo "  3) beam=10, length_penalty=0.5"
echo "  4) beam=10, length_penalty=0.7"
echo "  5) beam=10, length_penalty=1.3"
echo "  6) Run options 1-5, in order"
echo "  7) beam=5, length_penalty=1.5   (extended -- 1.3 was still climbing, testing further)"
echo "  8) beam=5, length_penalty=1.7   (extended)"
echo "  9) beam=5, length_penalty=2.0   (extended)"
echo " 10) Run the extended set (7-9), in order"
echo ""
read -p "Choose an option [1-10]: " choice

case "$choice" in
    1|2|3|4|5|7|8|9)
        run_option "$choice"
        ;;
    6)
        for i in 1 2 3 4 5; do
            run_option "$i"
        done
        ;;
    10)
        for i in 7 8 9; do
            run_option "$i"
        done
        ;;
    *)
        echo "!! Invalid choice: '$choice' -- expected 1-10."
        exit 1
        ;;
esac

echo ""
echo "############################################################"
echo "# Sweep complete. Files saved under: $OUT_DIR"
echo "############################################################"
ls -la "$OUT_DIR" 2>/dev/null
