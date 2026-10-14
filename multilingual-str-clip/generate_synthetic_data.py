"""
generate_synthetic_data.py
Generates synthetic word-crop images to supplement a real training
set. Two complementary modes, both usable together:

  1. STYLE AUGMENTATION (always on): re-renders every word already in your
     real --train_ann in multiple fonts/sizes/colors/distortions. This adds
     visual diversity for words the model already has labels for, directly
     targeting "not enough visual variation per word to generalize."

  2. VOCABULARY EXPANSION (optional, --corpus_file): if you can supply a
     plain text file of additional real words in your target script (one per line, UTF-8
     — e.g. from a dictionary or corpus you have access to), those get
     rendered too, expanding word/character-context diversity beyond what
     your real labeled set covers.

REQUIRES real font files (.ttf/.otf) for your target script — point
--fonts_dir at a folder containing them. Google Fonts' Noto family covers
essentially every script this project supports (Noto Sans Telugu, Noto Sans
Devanagari for Hindi, Noto Sans Tamil, Noto Sans Malayalam, etc.) — search
fonts.google.com for your language. This script does not download fonts
itself. Correct rendering of complex scripts (conjuncts, ligatures, matras)
requires Pillow to have been built with raqm/harfbuzz support —
check with:
    python -c "from PIL import features; print(features.check('raqm'))"
If that prints False, `pip install --upgrade --force-reinstall Pillow`
usually pulls a wheel with raqm bundled (most official Pillow wheels do).

Usage:
    python generate_synthetic_data.py \
        --train_ann data/train.tsv \
        --fonts_dir ./fonts \
        --out_dir data/synthetic \
        --samples_per_word 6 \
        --corpus_file extra_words.txt   # optional

Output: data/synthetic/images/*.jpg + data/synthetic/synthetic.tsv
(same image_path<TAB>label format as the rest of this project). Combine
with your real data by concatenating TSVs:
    cat data/train.tsv data/synthetic/synthetic.tsv > data/train_augmented.tsv
and point --train_ann at the combined file — update --image_root to a
common parent directory that contains both the real and synthetic image
folders (or copy synthetic images under your existing image_root).
"""
import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter, features


def load_words(train_ann, corpus_file=None, exclude_ann=None):
    base_words = set()
    with open(train_ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                base_words.add(parts[1].strip())

    words = set(base_words)
    if corpus_file:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)

    excluded_count = 0
    if exclude_ann:
        exclude_words = set()
        for path in exclude_ann:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2 and parts[1].strip():
                        exclude_words.add(parts[1].strip())
        before = len(words)
        words -= exclude_words
        excluded_count = before - len(words)
        base_words -= exclude_words

    # base_words is train_ann-only (post-exclusion) -- returned separately so
    # callers can compute --length_boost's avg_len from the ORIGINAL
    # vocabulary alone. If avg_len were computed over words (which includes
    # --corpus_file additions), adding many long corpus words would pull the
    # average length up and silently REDUCE the length_boost factor -- and
    # therefore the sample count -- given to every existing word, undoing
    # exactly the effect a length_boost retrain was tuned to produce. See the
    # --corpus_file help text for the concrete numbers that exposed this.
    return sorted(words), excluded_count, sorted(base_words)


def find_fonts(fonts_dir):
    exts = (".ttf", ".otf", ".ttc")
    fonts = [
        os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
        if f.lower().endswith(exts)
    ]
    return fonts


BG_PALETTE = [
    (255, 255, 255), (245, 245, 245), (230, 230, 230), (20, 20, 20),
    (250, 240, 220), (210, 225, 235), (30, 30, 40), (200, 200, 200),
]


def contrasting_fg(bg):
    brightness = sum(bg) / 3
    if brightness > 128:
        return tuple(random.randint(0, 60) for _ in range(3))
    return tuple(random.randint(195, 255) for _ in range(3))


def render_word(word, font_path, font_size, pad=14):
    font = ImageFont.truetype(font_path, font_size)
    # Measure with a scratch image first so the canvas is sized to the
    # actual rendered text (letters/matras vary a lot in height/width).
    scratch = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bg = random.choice(BG_PALETTE)
    fg = contrasting_fg(bg)
    canvas = Image.new("RGB", (text_w + 2 * pad, text_h + 2 * pad), bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad - bbox[0], pad - bbox[1]), word, font=font, fill=fg)
    return canvas


def augment(image):
    # Mirrors the real-data augmentation in dataset.py so synthetic and
    # real images end up in a visually similar distribution after
    # preprocessing — rotation, blur, and a bit of noise.
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-4, 4), expand=True,
                              fillcolor=random.choice(BG_PALETTE))
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    if random.random() < 0.3:
        # Cheap noise: blend in a random-noise layer at low opacity.
        import numpy as np
        arr = np.array(image).astype("int16")
        noise = np.random.randint(-20, 20, arr.shape, dtype="int16")
        arr = (arr + noise).clip(0, 255).astype("uint8")
        image = Image.fromarray(arr)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", required=True, help="Real train.tsv — its words get style-augmented")
    ap.add_argument("--corpus_file", default=None,
                     help="Optional extra word list, one word per line in your target script (UTF-8), to expand vocabulary")
    ap.add_argument("--exclude_ann", nargs="*", default=None,
                     help="One or more TSV files (e.g. data/val.tsv data/test.tsv) whose LABEL TEXT must "
                          "never be rendered here, even if the same word also happens to appear in "
                          "--train_ann under a different image. Without this, a word that repeats across "
                          "your dataset (common for signage vocabulary like 'Shop'/'Store'/'Party') can "
                          "get heavily synthetic-augmented purely because it's in train_ann, giving the "
                          "model strong memorized exposure to that exact text before it ever sees your "
                          "val/test images — a real form of leakage that doesn't show up as an image-path "
                          "overlap. STRONGLY recommended: always pass your val.tsv and test.tsv here.")
    ap.add_argument("--fonts_dir", required=True, help="Folder containing .ttf/.otf font files for your target script")
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--samples_per_word", type=int, default=6,
                     help="Base number of rendered variants (font/size/color/distortion) per word")
    ap.add_argument("--length_boost", type=float, default=0.0,
                     help="Oversample longer words: each word gets "
                          "samples_per_word * (1 + length_boost * (len(word)/avg_len - 1)) variants, "
                          "rounded and floored at 1. 0 (default) = no boost, every word gets exactly "
                          "--samples_per_word. Try 0.5-1.0 if your model shows premature-EOS/truncation "
                          "specifically on longer words (check with a length-bucketed WRR/CRR breakdown "
                          "on your eval predictions) — real training data usually skews toward shorter "
                          "words, so the decoder sees proportionally less signal to 'keep going' at "
                          "deeper positions unless longer words are deliberately oversampled.")
    ap.add_argument("--length_boost_floor", action="store_true",
                     help="Modifies --length_boost so it only ever INCREASES samples for "
                          "above-average-length words, never decreases samples for "
                          "below-average-length words (the plain formula does both: it's a "
                          "linear reallocation, so pushing length_boost up to help long words "
                          "necessarily starves short/medium words of samples at the same time). "
                          "Off by default for backward compatibility with already-validated runs "
                          "(e.g. Hindi's checkpoints/hindi_lr1.5e-5_lb0.8_unfreeze8_ep150, Telugu's "
                          "checkpoints/telugu_spw12) — those used the plain linear formula and "
                          "should not be silently reinterpreted. Diagnosed for Telugu Aug 22 2026: "
                          "at length_boost=1.5 (avg word length 8.4 chars), a 6-char word's sample "
                          "count dropped from 9 (at length_boost=0.8) to 7, and that squeeze on "
                          "shorter words in the medium (6-9 char) bucket lined up with a measured "
                          "-4.1pp WRR regression there, even though long-word (10+ char) WRR rose "
                          "+8.9pp in the same run. This flag isolates the long-word gain from the "
                          "medium-word cost.")
    ap.add_argument("--font_size_min", type=int, default=28)
    ap.add_argument("--font_size_max", type=int, default=56)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not features.check("raqm"):
        print("!! WARNING: Pillow was NOT built with raqm/harfbuzz support — complex script "
              "features (conjuncts, matras, ligatures) will likely render incorrectly. "
              "Run: pip install --upgrade --force-reinstall Pillow")

    random.seed(args.seed)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    words, excluded_count, base_words = load_words(args.train_ann, args.corpus_file, args.exclude_ann)
    if args.exclude_ann:
        print(f"Excluded {excluded_count} word(s) that also appear in: {', '.join(args.exclude_ann)}")
    else:
        print("!! WARNING: --exclude_ann not set. If any word in --train_ann also appears as a label "
              "in your val/test set (common for repeated signage vocabulary), it will get synthetic-"
              "augmented anyway, which can silently leak into your evaluation numbers. Strongly "
              "recommend re-running with --exclude_ann data/val.tsv data/test.tsv.")
    print(f"Words to render: {len(words)}")

    # avg_len is deliberately computed from base_words (--train_ann only, NOT
    # --corpus_file additions) so that adding a corpus can never shift the
    # sample count given to existing words -- see load_words()'s docstring
    # note. When --corpus_file is unset, base_words == words and this is
    # identical to the previous behavior.
    avg_len = sum(len(w) for w in base_words) / max(len(base_words), 1)
    if args.corpus_file and args.length_boost > 0:
        corpus_avg_len = sum(len(w) for w in words) / max(len(words), 1)
        print(f"  (base-vocabulary avg length: {avg_len:.1f} chars -- length_boost uses this, "
              f"NOT the corpus-inflated {corpus_avg_len:.1f})")

    def samples_for(word):
        if args.length_boost <= 0:
            return args.samples_per_word
        factor = 1 + args.length_boost * (len(word) / avg_len - 1)
        if args.length_boost_floor:
            factor = max(1.0, factor)
        return max(1, round(args.samples_per_word * factor))

    if args.length_boost > 0:
        shortest, longest = min(words, key=len), max(words, key=len)
        print(f"Length boost enabled (avg word length: {avg_len:.1f} chars) — e.g. "
              f"'{shortest}' ({len(shortest)} chars) gets {samples_for(shortest)} samples, "
              f"'{longest}' ({len(longest)} chars) gets {samples_for(longest)} samples.")

    fonts = find_fonts(args.fonts_dir)
    if not fonts:
        raise RuntimeError(
            f"No .ttf/.otf/.ttc font files found in {args.fonts_dir}. Download a font for your "
            f"target script (search fonts.google.com — the Noto family covers most scripts) "
            f"and point --fonts_dir at the folder containing it."
        )
    print(f"Fonts found: {len(fonts)} (in {args.fonts_dir})")

    tsv_path = os.path.join(args.out_dir, "synthetic.tsv")
    n_written = 0
    with open(tsv_path, "w", encoding="utf-8") as tsv_out:
        for word in words:
            for i in range(samples_for(word)):
                font_path = random.choice(fonts)
                font_size = random.randint(args.font_size_min, args.font_size_max)
                try:
                    img = render_word(word, font_path, font_size)
                except Exception as e:
                    print(f"  !! skipped '{word}' with font {os.path.basename(font_path)}: {e}")
                    continue
                img = augment(img)

                fname = f"images/syn_{n_written:06d}.jpg"
                img.convert("RGB").save(os.path.join(args.out_dir, fname), quality=92)
                tsv_out.write(f"{fname}\t{word}\n")
                n_written += 1

    print(f"\nWrote {n_written} synthetic images + labels to:")
    print(f"  {os.path.join(args.out_dir, 'images')}/")
    print(f"  {tsv_path}")
    print(f"\nCombine with real data:")
    print(f"  cat {args.train_ann} {tsv_path} > data/train_augmented.tsv")
    print(f"Then point train.py at the combined file, with --image_root set to a directory "
          f"that contains both your real images and {args.out_dir}/images "
          f"(e.g. symlink {args.out_dir}/images into your existing image_root, or keep the "
          f"synthetic image_path values as '{args.out_dir}/images/...' relative to a shared root).")
    print(f"\nBefore training on these: open a handful of the generated images and visually "
          f"confirm the script rendered correctly — conjuncts/matras/ligatures in particular "
          f"(this depends on your font files and Pillow's raqm support) — garbled synthetic "
          f"labels would actively hurt training.")


if __name__ == "__main__":
    main()
