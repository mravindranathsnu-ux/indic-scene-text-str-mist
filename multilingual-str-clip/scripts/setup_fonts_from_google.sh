#!/bin/bash
# setup_fonts_from_google.sh — the full fonts pipeline in one command:
# download every font supporting a language's script from the Google
# Fonts API, organize into fonts/<language>/, render every font, and
# build a contact sheet for review. No manual browsing, no zip downloads,
# no need to remember subset codes or retype the API key.
#
# ONE-TIME SETUP: save your free API key (get one at
# https://developers.google.com/fonts/docs/developer_api -> "Get a Key"):
#   echo "YOUR_KEY_HERE" > .google_fonts_api_key
# Run that from the project root. This file is read automatically below
# and should NOT be committed/shared -- it's just your key, in one place,
# instead of pasted into every command or script.
#
# Usage:
#   ./setup_fonts_from_google.sh <language>
#   ./setup_fonts_from_google.sh telugu
#   ./setup_fonts_from_google.sh punjabi              # auto-maps to subset "gurmukhi"
#   ./setup_fonts_from_google.sh <language> <subset>   # override the auto-mapped subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

# Language -> Google Fonts subset code. Several of these do NOT match the
# language name -- confirmed against the current Fonts API subset list.
declare -A SUBSET_MAP=(
    [telugu]=telugu
    [tamil]=tamil
    [kannada]=kannada
    [malayalam]=malayalam
    [bengali]=bengali
    [gujarati]=gujarati
    [punjabi]=gurmukhi
    [hindi]=devanagari
    [marathi]=devanagari
    [odia]=oriya
)

SUBSET_OVERRIDE="${2:-}"
if [ -n "$SUBSET_OVERRIDE" ]; then
    SUBSET="$SUBSET_OVERRIDE"
elif [ -n "${SUBSET_MAP[$LANGUAGE]:-}" ]; then
    SUBSET="${SUBSET_MAP[$LANGUAGE]}"
else
    echo "!! No known Google Fonts subset mapping for '$LANGUAGE'."
    echo "!! Pass it explicitly: $0 $LANGUAGE <subset>"
    echo "!! (check available subsets by running download_fonts_by_subset.py with any"
    echo "!! guess and --dry_run -- it prints the full known-subset list on a miss)"
    exit 1
fi

KEY_FILE="$PROJECT_ROOT/.google_fonts_api_key"
if [ ! -f "$KEY_FILE" ]; then
    echo "!! No stored API key found at $KEY_FILE"
    echo "!! One-time setup: get a free key at"
    echo "!!   https://developers.google.com/fonts/docs/developer_api  (click \"Get a Key\")"
    echo "!! then save it:"
    echo "!!   echo \"YOUR_KEY_HERE\" > $KEY_FILE"
    exit 1
fi
API_KEY="$(cat "$KEY_FILE" | tr -d '[:space:]')"
if [ -z "$API_KEY" ]; then
    echo "!! $KEY_FILE exists but is empty."
    exit 1
fi

RAW_DIR="incoming/fonts_${LANGUAGE}_raw"

echo "############################################################"
echo "# Fonts pipeline (Google Fonts API): $LANGUAGE  (subset: $SUBSET)"
echo "############################################################"

echo ""
echo "--- Step 1: preview which font families are available ---"
"$PYTHON" download_fonts_by_subset.py --subset "$SUBSET" --api_key "$API_KEY" \
    --out_dir "$RAW_DIR" --dry_run
DRYRUN_STATUS=$?
if [ $DRYRUN_STATUS -ne 0 ]; then
    echo ""
    echo "!! Preview failed (see above) -- nothing was downloaded. Fix the issue and re-run."
    exit 1
fi

echo ""
read -p ">>> Proceed with downloading these fonts? [y/N]: " confirm1
if [ "$confirm1" != "y" ] && [ "$confirm1" != "Y" ]; then
    echo "Aborted -- nothing downloaded."
    exit 0
fi

echo ""
echo "--- Step 2: download ---"
"$PYTHON" download_fonts_by_subset.py --subset "$SUBSET" --api_key "$API_KEY" \
    --out_dir "$RAW_DIR"
if [ $? -ne 0 ]; then
    echo "!! Download failed -- stopping before organize/verify."
    exit 1
fi

echo ""
echo "--- Step 3: preview organizing into fonts/$LANGUAGE/ ---"
"$PYTHON" organize_fonts.py --language "$LANGUAGE" --extracted_dir "$RAW_DIR" --dry_run
if [ $? -ne 0 ]; then
    echo ""
    echo "!! Organize preview failed (see above) -- stopping."
    exit 1
fi

echo ""
read -p ">>> Proceed with copying these fonts into fonts/$LANGUAGE/? [y/N]: " confirm2
if [ "$confirm2" != "y" ] && [ "$confirm2" != "Y" ]; then
    echo "Aborted -- downloaded to $RAW_DIR but not copied into fonts/$LANGUAGE/."
    exit 0
fi

echo ""
echo "--- Step 4: organize ---"
"$PYTHON" organize_fonts.py --language "$LANGUAGE" --extracted_dir "$RAW_DIR"
if [ $? -ne 0 ]; then
    echo "!! Organize failed -- stopping before verify."
    exit 1
fi

echo ""
echo "--- Step 5: render every font ---"
"$SCRIPT_DIR/03_verify_fonts.sh" "$LANGUAGE"

echo ""
echo "--- Step 6: build contact sheet ---"
"$PYTHON" make_font_contact_sheet.py --language "$LANGUAGE"

echo ""
echo "############################################################"
echo "# DONE"
echo "############################################################"
echo "Review: font_check_samples/${LANGUAGE}_contact_sheet.png"
echo "Once it looks good, regenerate synthetic data and retrain:"
echo "  cd scripts && FORCE_REGEN=1 ./run_${LANGUAGE}.sh"
