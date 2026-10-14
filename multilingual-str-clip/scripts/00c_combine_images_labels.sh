#!/bin/bash
# Step 0c (OPTIONAL): Combine separate train_images.txt/train_labels.txt
# (and val_/test_ equivalents) into this project's train.tsv/val.tsv/
# test.tsv format. Use this if your raw data ships as parallel image-list
# and label-list files rather than already-combined TSVs.
#
# Expects, for each split, these files already in place:
#   $DATA_DIR/<split>/<split>_images.txt   (one image filename/path per line)
#   $DATA_DIR/<split>/<split>_labels.txt   (one label per line, same order)
#
# Writes:
#   $DATA_DIR/<split>/<split>.tsv
#
# IMPORTANT — check your raw *_images.txt content before running, since
# the right prefix handling depends on what's already in there:
#   - Bare filenames (e.g. "img_0001.jpg")             -> set IMAGE_PREFIX="images/", STRIP_PREFIX=""
#   - Already full paths (e.g. "images/img_0001.jpg")  -> set IMAGE_PREFIX="",        STRIP_PREFIX=""
#   - Raw dataset's OWN internal structure baked in
#     (e.g. "train/images/img_0001.jpg", which would
#     duplicate part of --train_image_root if prefixed
#     further)                                          -> set STRIP_PREFIX="$split/images/" (default below)
# Default here assumes the third case (STRIP_PREFIX="$split/images/" per
# split, IMAGE_PREFIX="") since that's what this project has actually
# run into — check a few lines of your own *_images.txt against
# `ls $DATA_DIR/<split>/images/` first and adjust below if different.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 0c: Combine images.txt + labels.txt -> TSV for $LANGUAGE ==="
cd "$PROJECT_ROOT"

IMAGE_PREFIX=""   # prepended AFTER stripping — see comment block above

for split in train val test; do
    split_dir="$DATA_DIR/$split"
    images_txt="$split_dir/${split}_images.txt"
    labels_txt="$split_dir/${split}_labels.txt"
    out_tsv="$split_dir/${split}.tsv"
    strip_prefix="$split/images/"   # matches this project's actual raw-data case; adjust per your data

    if [ ! -f "$images_txt" ] || [ ! -f "$labels_txt" ]; then
        echo "  [$split] skipped — $images_txt or $labels_txt not found"
        continue
    fi

    "$PYTHON" combine_images_labels.py \
        --images_txt "$images_txt" \
        --labels_txt "$labels_txt" \
        --output "$out_tsv" \
        --image_prefix "$IMAGE_PREFIX" \
        --strip_prefix "$strip_prefix"
done

echo ""
echo "Done. Any split reported 'skipped' above had no matching *_images.txt/"
echo "*_labels.txt pair under $DATA_DIR/<split>/ — check filenames if unexpected."
echo ""
echo ">>> ACTION REQUIRED: spot-check the output TSVs before running sanity_check.py —"
echo ">>> e.g. 'head -3 $DATA_DIR/train/train.tsv' and confirm each image_path, when"
echo ">>> joined with --train_image_root, actually points at a real file:"
echo ">>>   ls \"\$(dirname \"$DATA_DIR/train/images/<first image_path from the TSV>\")\""
