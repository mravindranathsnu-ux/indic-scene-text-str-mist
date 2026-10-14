#!/bin/bash
# lexicon_rerank.sh — wrapper around lexicon_rerank_eval.py. Takes just a
# language and handles everything else: checkpoint path, test set, lexicon
# sources (train+val+IndicCorp), and per-language decode parameters.
#
# Per-language length_penalty is validated where we've actually run a
# decode sweep (Telugu, Tamil) and defaults to the untuned 1.0 everywhere
# else, since Tamil's own sweep found no gain from tuning it and we chose
# not to re-run a sweep for every other language on that basis.
#
# All other rerank parameters are fixed at the values validated during
# Telugu's optimization and confirmed not to regress on Tamil:
#   beam_size=5, max_edit_distance=1, suffix_guard_chars=1,
#   frequency tie-break DISABLED (found to be a net regression on Telugu).
#
# Usage:
#   ./scripts/lexicon_rerank.sh <language>
#   ./scripts/lexicon_rerank.sh punjabi
#
# Requires data/<language>/indiccorp_lexicon.tsv to already exist --
# run ./scripts/build_lexicon.sh <language> first if it doesn't.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

VALID_LANGUAGES="bengali gujarati hindi kannada malayalam marathi odia punjabi tamil telugu"

LANG_ARG="${1:-}"
if [ -z "$LANG_ARG" ]; then
    echo "Usage: $0 <language>"
    echo "  e.g. $0 punjabi"
    exit 1
fi

case " $VALID_LANGUAGES " in
    *" $LANG_ARG "*) ;;
    *)
        echo "!! Unknown language: '$LANG_ARG'"
        echo "!! Supported: $VALID_LANGUAGES"
        exit 1
        ;;
esac

# Per-language length_penalty: validated values for Telugu/Tamil, untuned
# default (1.0) for every other language (no decode sweep run for them).
case "$LANG_ARG" in
    telugu) LENGTH_PENALTY=1.3 ;;
    tamil)  LENGTH_PENALTY=1.0 ;;
    *)      LENGTH_PENALTY=1.0 ;;
esac

CKPT="$PROJECT_ROOT/checkpoints/$LANG_ARG/best.pt"
CLIP_CKPT="$PROJECT_ROOT/weights/ViT-B-16.pt"
TEST_ANN="$PROJECT_ROOT/data/$LANG_ARG/test/test.tsv"
TEST_IMG="$PROJECT_ROOT/data/$LANG_ARG/test/images"
TRAIN_ANN="$PROJECT_ROOT/data/$LANG_ARG/train/train.tsv"
VAL_ANN="$PROJECT_ROOT/data/$LANG_ARG/val/val.tsv"
INDICCORP_LEXICON="$PROJECT_ROOT/data/$LANG_ARG/indiccorp_lexicon.tsv"
OUT_CSV="$PROJECT_ROOT/results/${LANG_ARG}_lexicon_rerank.csv"

# Fail early and clearly rather than let the python script hit a missing
# lexicon file partway through -- same rationale as every other pipeline
# script's file-existence checks in this project.
MISSING=0
for f in "$CKPT" "$CLIP_CKPT" "$TEST_ANN" "$TRAIN_ANN" "$VAL_ANN" "$INDICCORP_LEXICON"; do
    if [ ! -f "$f" ]; then
        echo "!! Missing required file: $f"
        MISSING=1
    fi
done
if [ ! -d "$TEST_IMG" ]; then
    echo "!! Missing required directory: $TEST_IMG"
    MISSING=1
fi
if [ $MISSING -eq 1 ]; then
    if [ ! -f "$INDICCORP_LEXICON" ]; then
        echo "!! Hint: run ./scripts/build_lexicon.sh $LANG_ARG first to build the missing lexicon file."
    fi
    exit 1
fi

if [ "$LANG_ARG" = "telugu" ] || [ "$LANG_ARG" = "tamil" ]; then
    LP_NOTE="validated for $LANG_ARG"
else
    LP_NOTE="untuned default -- no decode sweep run for $LANG_ARG"
fi

echo "############################################################"
echo "# Lexicon-aware beam reranking for $LANG_ARG"
echo "# length_penalty=$LENGTH_PENALTY ($LP_NOTE)"
echo "# Lexicon sources: train.tsv, val.tsv, indiccorp_lexicon.tsv"
echo "# Output: $OUT_CSV"
echo "############################################################"

cd "$PROJECT_ROOT"
"$PYTHON" lexicon_rerank_eval.py \
    --ckpt "$CKPT" \
    --clip_ckpt "$CLIP_CKPT" \
    --test_ann "$TEST_ANN" \
    --test_image_root "$TEST_IMG" \
    --lexicon_ann "$TRAIN_ANN" "$VAL_ANN" "$INDICCORP_LEXICON" \
    --beam_size 5 \
    --length_penalty "$LENGTH_PENALTY" \
    --max_edit_distance 1 \
    --suffix_guard_chars 1 \
    --no_tie_break_by_frequency \
    --out_csv "$OUT_CSV" \
    --yes
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "############################################################"
    echo "# Done. Compare the printed \"before\" (top-1 beam) WRR/CRR against"
    echo "# your last known evaluate.py result for $LANG_ARG to confirm this"
    echo "# ran against the correct checkpoint before trusting the \"after\" number."
    echo "############################################################"
fi
exit $STATUS
