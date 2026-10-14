#!/bin/bash
# Step 0a (OPTIONAL): Convert plain character-list .txt files (one char
# per line — the common CRNN/CTC vocab format) into the JSON vocab files
# vocab.py expects. Skip this script entirely if vocab/<language>_vocab.json
# already exists.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 0a: Convert char-list files -> vocab JSON ==="
cd "$PROJECT_ROOT"

if [ ! -d "$CHAR_LISTS_DIR" ]; then
    echo "No $CHAR_LISTS_DIR directory found — nothing to convert."
    echo "(This step is only needed if you have raw .txt char-list files to convert;"
    echo " if vocab/${LANGUAGE}_vocab.json already exists, skip straight to step 01.)"
    exit 0
fi

"$PYTHON" convert_char_list.py --input_dir "$CHAR_LISTS_DIR" --output_dir "$VOCAB_DIR"
echo "Vocab files written to $VOCAB_DIR/"
