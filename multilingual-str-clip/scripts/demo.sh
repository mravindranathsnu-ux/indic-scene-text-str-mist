#!/bin/bash
# demo.sh — for live demos: grab a single image (from a URL, live, or a
# local file) and run it through one language's trained model, printing
# both the full technical output and a clean, large "punchline" result.
#
# Usage:
#   ./demo.sh telugu https://example.com/some_signboard_photo.jpg
#   ./demo.sh telugu /path/to/local/image.png
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

IMAGE_ARG="${2:-}"
if [ -z "$IMAGE_ARG" ]; then
    echo "Usage: $0 <language> <image_url_or_local_path>"
    echo "Example: $0 telugu https://example.com/signboard.jpg"
    exit 1
fi

DEMO_DIR="$PROJECT_ROOT/demo"
mkdir -p "$DEMO_DIR"

if [[ "$IMAGE_ARG" =~ ^https?:// ]]; then
    EXT="${IMAGE_ARG##*.}"
    EXT="${EXT%%\?*}"   # strip any query string caught in the extension guess
    EXT="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"
    case "$EXT" in
        jpg|jpeg|png|bmp|tiff|webp) ;;
        *) EXT="jpg" ;;
    esac
    IMAGE_PATH="$DEMO_DIR/${LANGUAGE}_demo.$EXT"
    echo "Downloading image..."
    echo "  from: $IMAGE_ARG"
    echo "  to:   $IMAGE_PATH"
    if ! curl -sL --fail "$IMAGE_ARG" -o "$IMAGE_PATH"; then
        echo "!! Download failed (bad URL, network issue, or server error). Nothing to run."
        exit 1
    fi
    if [ ! -s "$IMAGE_PATH" ]; then
        echo "!! Downloaded file is empty — the URL likely didn't point directly at an image file."
        rm -f "$IMAGE_PATH"
        exit 1
    fi
else
    IMAGE_PATH="$IMAGE_ARG"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "!! Local file not found: $IMAGE_PATH"
        exit 1
    fi
    # Resolve to an absolute path now, while we're still in the caller's
    # original working directory -- a relative path like ../photo.jpg would
    # otherwise silently resolve against the wrong directory once we cd
    # below (needed so inference.py, which lives in the project root, can
    # be found as a relative path).
    IMAGE_PATH="$(cd "$(dirname "$IMAGE_PATH")" && pwd)/$(basename "$IMAGE_PATH")"
fi

cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo " LIVE DEMO: $LANGUAGE"
echo "============================================================"
echo " Image: $IMAGE_PATH"
echo "------------------------------------------------------------"

OUTPUT=$("$PYTHON" inference.py \
    --ckpt "$OUT_DIR/best.pt" \
    --clip_ckpt "$CLIP_CKPT" \
    --image "$IMAGE_PATH" \
    --decode beam --beam_size "$BEAM_SIZE" --length_penalty "$LENGTH_PENALTY" \
    --yes)
echo "$OUTPUT"

PREDICTION=$(echo "$OUTPUT" | grep -F "$IMAGE_PATH" | tail -1 | cut -f2)

echo ""
echo "############################################################"
if [ -n "$PREDICTION" ]; then
    echo "#  PREDICTED TEXT ($LANGUAGE):  $PREDICTION"
else
    echo "#  No prediction line found — check the output above for an error."
fi
echo "############################################################"

GALLERY_PATH="$PROJECT_ROOT/demo/gallery.html"
"$PYTHON" "$PROJECT_ROOT/append_to_demo_gallery.py" \
    --language "$LANGUAGE" \
    --image_path "$IMAGE_PATH" \
    --prediction "$PREDICTION" \
    --gallery_path "$GALLERY_PATH" \
    --project_root "$PROJECT_ROOT"

echo ""
echo ">>> Gallery: $GALLERY_PATH"
echo ">>> Open it once at the start of your demo (in a browser tab) — press refresh"
echo ">>> in the browser after each new result to bring it to the top."
