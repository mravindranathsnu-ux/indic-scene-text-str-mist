#!/bin/bash
# setup_fonts.sh — runs the fonts workflow (steps 2-5 of FONTS_WORKFLOW.md)
# in one command: preview -> confirm -> organize -> render samples ->
# contact sheet. Step 1 (placing the archive in incoming/) must already be
# done manually before running this.
#
# Usage:
#   ./setup_fonts.sh <language>
#   ./setup_fonts.sh <language> /path/to/font_archive.zip   (explicit archive path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

ARCHIVE_ARG="${2:-}"

echo "############################################################"
echo "# Fonts setup: $LANGUAGE"
echo "############################################################"

ORGANIZE_ARGS=(--language "$LANGUAGE")
if [ -n "$ARCHIVE_ARG" ]; then
    ORGANIZE_ARGS+=(--archive "$ARCHIVE_ARG")
fi

echo ""
echo "--- Step 2: preview extraction ---"
"$PYTHON" organize_fonts.py "${ORGANIZE_ARGS[@]}" --dry_run
DRYRUN_STATUS=$?
if [ $DRYRUN_STATUS -ne 0 ]; then
    echo ""
    echo "!! Preview failed (see above) -- nothing was copied. Fix the issue (e.g. place the"
    echo "!! archive at incoming/fonts_${LANGUAGE}.zip, or pass its path as the 2nd argument"
    echo "!! to this script) and re-run."
    exit 1
fi

echo ""
read -p ">>> Proceed with copying these fonts into fonts/$LANGUAGE/? [y/N]: " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted -- no fonts copied."
    exit 0
fi

echo ""
echo "--- Step 3: copy fonts ---"
"$PYTHON" organize_fonts.py "${ORGANIZE_ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "!! Copy step failed -- stopping before rendering/contact sheet."
    exit 1
fi

echo ""
echo "--- Step 4: render sample word with every font ---"
"$SCRIPT_DIR/03_verify_fonts.sh" "$LANGUAGE"

echo ""
echo "--- Step 5: combine into contact sheet ---"
"$PYTHON" make_font_contact_sheet.py --language "$LANGUAGE"

echo ""
echo "############################################################"
echo "# DONE"
echo "############################################################"
echo "Review: font_check_samples/${LANGUAGE}_contact_sheet.png"
echo "Once it looks good, proceed with: ./run_${LANGUAGE}.sh"
