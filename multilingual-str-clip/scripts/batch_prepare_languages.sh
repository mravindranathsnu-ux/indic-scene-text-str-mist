#!/bin/bash
# batch_prepare_languages.sh — for each language given (or the default list
# below): organizes images from incoming/<language>.*, builds train/val/test
# TSVs, and runs the duplicate/leakage check. Does NOT touch fonts, synthetic
# data, sanity checks, or training — safe to run while another language is
# training on the GPU, since none of this needs it.
#
# Usage:
#   ./batch_prepare_languages.sh bengali gujarati punjabi odia marathi
#   ./batch_prepare_languages.sh                      (uses the default list below)
#
# After this finishes, review each language's train/val/test TSV counts and
# duplicate-check output above, install its fonts manually, then run
# ./run_<language>.sh for it (which will skip the now-already-done organize/
# 00c/duplicate-check steps automatically where it can).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_LANGUAGES=(bengali gujarati punjabi odia marathi)
if [ "$#" -gt 0 ]; then
    LANGUAGES=("$@")
else
    LANGUAGES=("${DEFAULT_LANGUAGES[@]}")
fi

RESULTS=()

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Preparing: $lang"
    echo "############################################################"

    echo ""
    echo "--- organize images (incoming/${lang}.* -> data/${lang}/) ---"
    if ! python3 "$PROJECT_ROOT/organize_language_zip.py" --language "$lang" --project_root "$PROJECT_ROOT"; then
        echo "!! organize failed for $lang — check that incoming/${lang}.{tar.gz,zip,tgz,tar} exists."
        RESULTS+=("$lang: FAILED (organize)")
        continue
    fi

    echo ""
    echo "--- 00c: combine images.txt + labels.txt into TSVs ---"
    if ! "$SCRIPT_DIR/00c_combine_images_labels.sh" "$lang"; then
        echo "!! 00c failed for $lang"
        RESULTS+=("$lang: FAILED (00c)")
        continue
    fi

    if [ ! -f "$PROJECT_ROOT/data/$lang/train/train.tsv" ] || [ ! -f "$PROJECT_ROOT/data/$lang/test/test.tsv" ]; then
        echo "!! No train.tsv/test.tsv for $lang after 00c — skipping duplicate check."
        echo "   (raw *_images.txt/*_labels.txt may be missing or misnamed — see 00c's output above)"
        RESULTS+=("$lang: NO DATA after 00c")
        continue
    fi

    echo ""
    echo "--- 05: duplicate/leakage check ---"
    if ! "$SCRIPT_DIR/05_check_duplicates.sh" "$lang"; then
        echo "!! duplicate check errored for $lang"
        RESULTS+=("$lang: duplicate check ERRORED")
        continue
    fi

    RESULTS+=("$lang: OK — organized, TSVs built, duplicate check ran (review output above)")
done

echo ""
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Next, for each language above:"
echo "  1. Review its train/val/test TSV pair counts and duplicate-check result (scroll up)."
echo "  2. Install its fonts manually: fonts/<language>/*.ttf"
echo "  3. Once Hindi's training finishes and the GPU is free: ./run_<language>.sh"
