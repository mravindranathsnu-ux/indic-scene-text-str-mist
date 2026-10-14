"""
verify_fonts.py
Quick visual check that a language's fonts render correctly (correct
complex-script shaping — conjuncts, matras, ligatures) BEFORE running
generate_synthetic_data.py at scale on them. A font/Pillow combination
that silently mis-shapes text would poison your synthetic data with
garbled labels.

Usage:
    python verify_fonts.py --fonts_dir fonts/hindi --language hindi \
        --test_word "भारत" --out_dir font_check_samples

    python verify_fonts.py --fonts_dir fonts/tamil --language tamil \
        --test_word "தமிழ்" --out_dir font_check_samples

If --test_word is omitted, a default per-language sample word (containing
a conjunct/matra where possible) is used.
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont, features

# One example word per supported language, chosen to include a conjunct/
# matra/ligature so shaping problems are visible, not just plain glyphs.
DEFAULT_TEST_WORDS = {
    "telugu": "కొండాపూర్",
    "hindi": "क्षत्रिय",
    "devanagari": "क्षत्रिय",
    "marathi": "महाराष्ट्र",
    "tamil": "தமிழ்நாடு",
    "malayalam": "സ്വാഗതം",
    "kannada": "ಕರ್ನಾಟಕ",
    "bengali": "বাংলাদেশ",
    "gujarati": "ગુજરાત",
    "gurmukhi": "ਪੰਜਾਬ",
    "punjabi": "ਪੰਜਾਬ",
    "odia": "ଓଡ଼ିଶା",
    "english": "Hello",
    "latin": "Hello",
}


def find_fonts(fonts_dir):
    exts = (".ttf", ".otf", ".ttc")
    return [
        os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
        if f.lower().endswith(exts)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts_dir", required=True)
    ap.add_argument("--language", required=True,
                     help="Used only to pick a default --test_word if you don't supply one")
    ap.add_argument("--test_word", default=None)
    ap.add_argument("--font_size", type=int, default=48)
    ap.add_argument("--out_dir", default="font_check_samples")
    args = ap.parse_args()

    if not features.check("raqm"):
        print("!! WARNING: Pillow was NOT built with raqm/harfbuzz support — complex script "
              "shaping will likely be WRONG regardless of font quality. "
              "Run: pip install --upgrade --force-reinstall Pillow")
        print("   Continuing anyway so you can see how bad it looks without raqm.\n")

    word = args.test_word or DEFAULT_TEST_WORDS.get(args.language.lower())
    if not word:
        raise ValueError(
            f"No default test word for language '{args.language}' — pass --test_word explicitly."
        )

    fonts = find_fonts(args.fonts_dir)
    if not fonts:
        raise RuntimeError(f"No .ttf/.otf/.ttc files found in {args.fonts_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Testing {len(fonts)} font(s) with word: {word!r}")

    ok_count, fail_count = 0, 0
    for font_path in fonts:
        name = os.path.basename(font_path)
        try:
            font = ImageFont.truetype(font_path, args.font_size)
            scratch = Image.new("RGB", (10, 10))
            bbox = ImageDraw.Draw(scratch).textbbox((0, 0), word, font=font)
            w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20

            img = Image.new("RGB", (max(w, 50), max(h, 50)), "white")
            draw = ImageDraw.Draw(img)
            draw.text((10 - bbox[0], 10 - bbox[1]), word, font=font, fill="black")

            out_path = os.path.join(args.out_dir, f"{args.language}_{os.path.splitext(name)[0]}.png")
            img.save(out_path)
            ok_count += 1
        except Exception as e:
            print(f"  !! FAIL {name:40s} -> {e}")
            fail_count += 1

    print(f"\n{ok_count}/{len(fonts)} font(s) rendered without error -> {args.out_dir}/")
    if fail_count:
        print(f"{fail_count} font(s) FAILED to render at all (see above) -- those are unusable, "
              f"don't wait on them for the visual check below.")

    print(f"\nOpen the PNGs in {args.out_dir}/ and check: does '{word}' render legibly, "
          f"with conjuncts/matras/ligatures correctly formed (not separate floating marks "
          f"or missing/boxed glyphs)? Only use fonts that pass this check with "
          f"generate_synthetic_data.py.")


if __name__ == "__main__":
    main()
