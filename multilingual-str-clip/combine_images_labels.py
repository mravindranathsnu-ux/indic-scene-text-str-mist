"""
combine_images_labels.py
Combines a separate image-list file and label-list file (same line count,
line i of one corresponds to line i of the other — a common raw format,
e.g. CRNN-style train_images.txt / train_labels.txt) into the
image_path<TAB>label TSV format this project's dataset.py expects.

Usage:
    python combine_images_labels.py \
        --images_txt data/telugu/train/train_images.txt \
        --labels_txt data/telugu/train/train_labels.txt \
        --output data/telugu/train/train.tsv

If train_images.txt contains bare filenames (e.g. "img_0001.jpg") rather
than paths already relative to that split's images/ folder, use
--image_prefix to prepend a folder, e.g. --image_prefix images/ turns
"img_0001.jpg" into "images/img_0001.jpg" in the output TSV.
"""
import argparse
import os


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        # Only strip the trailing newline, not internal/trailing spaces —
        # a label could legitimately end in a space-adjacent punctuation
        # mark, and stripping too aggressively would silently corrupt it.
        return [line.rstrip("\n").rstrip("\r") for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_txt", required=True, help="One image path/filename per line")
    ap.add_argument("--labels_txt", required=True, help="One label per line, same order as --images_txt")
    ap.add_argument("--output", required=True, help="Output TSV path (image_path<TAB>label)")
    ap.add_argument("--image_prefix", default="",
                     help="Optional folder to PREPEND to each image path, e.g. 'images/' if "
                          "--images_txt has bare filenames not yet relative to an images/ subfolder")
    ap.add_argument("--strip_prefix", default="",
                     help="Optional prefix to REMOVE from each image path first, e.g. 'train/images/' "
                          "if your raw *_images.txt already contains a path like "
                          "'train/images/img_0001.jpg' that duplicates part of --train_image_root "
                          "once combined (a common issue when a raw dataset's own internal folder "
                          "structure is baked into the image list). Applied BEFORE --image_prefix.")
    ap.add_argument("--skip_blank", action="store_true", default=True,
                     help="Skip pairs where either the image path or label is blank after stripping (default on)")
    args = ap.parse_args()

    images = read_lines(args.images_txt)
    labels = read_lines(args.labels_txt)

    if len(images) != len(labels):
        raise ValueError(
            f"Line count mismatch: {args.images_txt} has {len(images)} lines, "
            f"{args.labels_txt} has {len(labels)} lines. They must correspond 1:1 — "
            f"fix the mismatch before combining (a common cause is a trailing blank "
            f"line in only one of the two files)."
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written, skipped_blank, skipped_tab = 0, 0, 0
    with open(args.output, "w", encoding="utf-8") as out:
        for img, label in zip(images, labels):
            img_stripped = img
            if args.strip_prefix and img_stripped.startswith(args.strip_prefix):
                img_stripped = img_stripped[len(args.strip_prefix):]
            img_path = args.image_prefix + img_stripped if args.image_prefix else img_stripped

            if args.skip_blank and (not img_path.strip() or not label.strip()):
                skipped_blank += 1
                continue

            # A literal tab in either field would corrupt the TSV structure
            # (dataset.py splits on the FIRST tab only, so a tab inside the
            # image path specifically would misalign the split — labels
            # containing a tab are vanishingly unlikely but checked too).
            if "\t" in img_path or "\t" in label:
                skipped_tab += 1
                continue

            out.write(f"{img_path}\t{label}\n")
            written += 1

    print(f"{args.images_txt} + {args.labels_txt} -> {args.output}")
    print(f"  {written} pairs written"
          + (f", {skipped_blank} blank pair(s) skipped" if skipped_blank else "")
          + (f", {skipped_tab} pair(s) with a literal tab skipped" if skipped_tab else ""))


if __name__ == "__main__":
    main()
