#!/bin/bash
# build_lexicon.sh — wrapper around build_lexicon_from_indiccorp.py.
# Builds a word-frequency lexicon from AI4Bharat's IndicCorpV2 for the
# given language, writing to data/<language>/indiccorp_lexicon.tsv.
#
# IMPORTANT CAVEAT (same as the underlying Python script): this is
# general-domain news/web text, not signage-specific vocabulary. Compare
# Fixed-vs-Broken against your existing lexicon before trusting it's
# actually better — a bigger lexicon is not automatically a better one.
#
# Usage:
#   ./scripts/build_lexicon.sh <language>
#   ./scripts/build_lexicon.sh telugu
#   ./scripts/build_lexicon.sh all      # all 10 languages, one run, unattended --
#                                       # checks the 'datasets' dependency once up
#                                       # front, then no further prompts; a failure
#                                       # on one language does NOT stop the rest,
#                                       # so one bad network moment overnight
#                                       # doesn't cost you the other 9 languages.
#
# Env var overrides:
#   INDICCORP_MAX_ROWS       sentences to stream before stopping (default 500000)
#   INDICCORP_MIN_FREQUENCY  drop words seen fewer than this many times (default 2)
#   INDICCORP_MIN_WORD_LEN   drop words shorter than this many characters (default 2 --
#                            single-character tokens from corpus tokenization noise were
#                            found matching against unrelated, much longer ground-truth
#                            words during Punjabi lexicon reranking; excluding them here
#                            is safer than leaving the underlying script's own default of 1)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

VALID_LANGUAGES="bengali gujarati hindi kannada malayalam marathi odia punjabi tamil telugu"

LANG_ARG="${1:-}"
if [ -z "$LANG_ARG" ]; then
    echo "Usage: $0 <language>"
    echo "  e.g. $0 telugu"
    echo "  or:  $0 all      (all 10 languages, unattended)"
    exit 1
fi

if [ "$LANG_ARG" != "all" ]; then
    case " $VALID_LANGUAGES " in
        *" $LANG_ARG "*) ;;
        *)
            echo "!! Unknown language: '$LANG_ARG'"
            echo "!! Supported: $VALID_LANGUAGES"
            echo "!! Or use: $0 all"
            exit 1
            ;;
    esac
fi

MAX_ROWS="${INDICCORP_MAX_ROWS:-500000}"
MIN_FREQUENCY="${INDICCORP_MIN_FREQUENCY:-2}"
MIN_WORD_LEN="${INDICCORP_MIN_WORD_LEN:-2}"

# Checked ONCE up front, regardless of single-language or 'all' mode, so
# 'all' mode never stalls on an interactive prompt partway through with
# no one there to answer it.
"$PYTHON" -c "import datasets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "The 'datasets' package isn't installed yet."
    read -p "Install it now with pip? [y/N]: " install_confirm
    if [ "$install_confirm" = "y" ] || [ "$install_confirm" = "Y" ]; then
        "$PYTHON" -m pip install datasets
        if [ $? -ne 0 ]; then
            echo "!! pip install failed -- install 'datasets' manually and re-run this script."
            exit 1
        fi
    else
        echo "Aborted -- 'datasets' is required for this script."
        exit 1
    fi
fi

build_one() {
    local lang="$1"
    local out_path="$PROJECT_ROOT/data/$lang/indiccorp_lexicon.tsv"

    echo "############################################################"
    echo "# Building IndicCorpV2 lexicon for $lang"
    echo "# max_rows=$MAX_ROWS  min_frequency=$MIN_FREQUENCY  min_word_len=$MIN_WORD_LEN"
    echo "# Output: $out_path"
    echo "############################################################"

    cd "$PROJECT_ROOT"
    "$PYTHON" build_lexicon_from_indiccorp.py \
        --language "$lang" \
        --max_rows "$MAX_ROWS" \
        --min_frequency "$MIN_FREQUENCY" \
        --min_word_len "$MIN_WORD_LEN" \
        --out "$out_path"
    return $?
}

if [ "$LANG_ARG" = "all" ]; then
    echo "############################################################"
    echo "# Building IndicCorpV2 lexicons for ALL $(echo $VALID_LANGUAGES | wc -w) languages, unattended"
    echo "# A failure on one language will NOT stop the rest."
    echo "############################################################"
    echo ""

    declare -a SUCCEEDED=()
    declare -a FAILED=()

    for lang in $VALID_LANGUAGES; do
        build_one "$lang"
        if [ $? -eq 0 ]; then
            SUCCEEDED+=("$lang")
        else
            echo "!! Build failed for '$lang' -- continuing to the next language."
            FAILED+=("$lang")
        fi
        echo ""
    done

    echo "############################################################"
    echo "# ALL-LANGUAGES BUILD SUMMARY"
    echo "############################################################"
    echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
    echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo ""
        echo "To retry just the failed ones:"
        for lang in "${FAILED[@]}"; do
            echo "  $0 $lang"
        done
    fi
    echo ""
    echo "Reminder for every lexicon built above: this is general-domain text,"
    echo "not signage-specific. Compare Fixed-vs-Broken against each language's"
    echo "existing lexicon before trusting it helps -- a bigger lexicon is not"
    echo "automatically a better one."
    echo "############################################################"

    [ ${#FAILED[@]} -eq 0 ]
    exit $?
fi

# Single-language mode
build_one "$LANG_ARG"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    OUT_PATH="$PROJECT_ROOT/data/$LANG_ARG/indiccorp_lexicon.tsv"
    echo ""
    echo "############################################################"
    echo "# Done. To use this lexicon, add it alongside (not instead of)"
    echo "# your existing train/val TSVs in --lexicon_ann, e.g.:"
    echo "#"
    echo "#   --lexicon_ann data/$LANG_ARG/train/train.tsv data/$LANG_ARG/val/val.tsv \\"
    echo "#                 $OUT_PATH"
    echo "#"
    echo "# Compare Fixed-vs-Broken against your current lexicon before trusting"
    echo "# this helps -- a bigger lexicon is not automatically a better one."
    echo "############################################################"
fi
exit $STATUS
