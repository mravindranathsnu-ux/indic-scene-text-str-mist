#!/bin/bash
# Step 3 (OPTIONAL — only needed if generating synthetic data in step 04):
# Render a stress-test word with every font in fonts/<language>/ and
# report which ones fail to load. STILL REQUIRES a human to open the
# output PNGs and visually confirm correct shaping (conjuncts/matras/
# ligatures) before trusting a font — this script can't verify that part
# automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 3: Verify fonts for $LANGUAGE ==="
cd "$PROJECT_ROOT"

if [ ! -d "$FONTS_DIR" ]; then
    echo "No $FONTS_DIR directory found — skipping (only needed for synthetic data generation)."
    exit 0
fi

"$PYTHON" verify_fonts.py --fonts_dir "$FONTS_DIR" --language "$LANGUAGE" --out_dir "$FONT_CHECK_OUT_DIR"

echo ""
echo ">>> ACTION REQUIRED: open the PNGs in $FONT_CHECK_OUT_DIR/ and visually"
echo ">>> confirm the script rendered correctly before running step 04."
