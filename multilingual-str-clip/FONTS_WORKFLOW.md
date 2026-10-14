# Fonts Workflow — Reference

Complete sequence for setting up fonts for any language, from a downloaded
zip through visual verification. Replace `<language>` with the actual
language name (e.g. `punjabi`) in every command.

All commands assume you're starting from the `multilingual-str-clip/`
project root unless otherwise noted.

---

## 1. Place the font archive in `incoming/`

```bash
cp ~/Downloads/<whatever_you_downloaded>.zip incoming/fonts_<language>.zip
```

Works with `.zip`, `.tar.gz`, `.tgz`, or `.tar` — `organize_fonts.py`
auto-detects the type. The exact filename doesn't matter as long as it's
named `fonts_<language>.<ext>`, since that's what gets auto-detected in
the next step. You can also skip this naming convention entirely and
point directly at the file with `--archive <path>` instead (see step 2).

---

## 2. Preview the extraction (touches nothing)

```bash
python organize_fonts.py --language <language> --dry_run
```

Extracts the archive to a staging folder under `incoming/`, recursively
finds every font file (`.ttf`/`.otf`/`.ttc`) regardless of how deeply
nested it is — handles the common case of Google Fonts downloads, where
each family sits in its own subfolder, often with a further `static/`
subfolder inside that. Non-font files (`OFL.txt`, `README.txt`) are
skipped automatically. Reports what it found, grouped by family/folder,
and checks for filename collisions before anything is copied.

If you didn't use the `fonts_<language>.<ext>` naming convention:
```bash
python organize_fonts.py --language <language> --archive /path/to/your/file.zip --dry_run
```

---

## 3. Actually copy the fonts

```bash
python organize_fonts.py --language <language>
```

Same as step 2, minus `--dry_run` — copies every font found, flat, into
`fonts/<language>/`.

---

## 4. Render a sample word with every font

```bash
cd scripts
./03_verify_fonts.sh <language>
cd ..
```

Renders each font's take on a representative test word to
`font_check_samples/<language>/*.png` — one image per font.

---

## 5. Combine all rendered samples into one scannable grid

```bash
python make_font_contact_sheet.py --language <language>
```

Produces `font_check_samples/<language>_contact_sheet.png` — every font
laid out in a labeled grid (6 columns by default; use `--cols N` to
change). Lets you review 100+ fonts in one glance instead of opening each
PNG individually.

---

## 6. Review the contact sheet

Open it and skim for:
- **Boxes / tofu** — missing glyph, font doesn't support this script properly
- **Disconnected or floating marks** — vowel signs/diacritics rendering separately from the base letter
- **Garbled shapes** — malformed glyph rendering
- **Obviously wrong script** — rare, but worth a glance

Faint/thin-looking cells (Thin, ExtraLight weights) are normal, not a
defect — don't mistake low visual weight for a rendering bug.

Only open individual PNGs (`font_check_samples/<language>/*.png`) for
fonts that actually look suspicious on the sheet — no need to check every
one individually if the sheet looks clean.

---

## 7. Once fonts pass review

Fonts are the last manual step before the pipeline can run unattended:

```bash
cd scripts
./run_<language>.sh
```

This picks up automatically from wherever the pipeline is up to for that
language (organize/combine/duplicate-check steps are skipped if already
done, per the pre-flight checklist).

---

## Worth knowing: font count is a variable, not just "more is better"

More **genuinely distinct typefaces** (different families) helps the
model generalize to varied real-world signage. More **weight/width
variants of the same family** (Bold, Light, Condensed, Expanded, etc.)
adds much less — they share the same underlying letterforms. If one
language ends up with a dramatically different total font count than the
others (e.g. a large multi-family Google Fonts bundle vs. a small curated
set), note it — it's a legitimate variable to consider if that
language's results look unusual compared to the rest, the same way
real/synthetic data volume is already tracked per language.

```bash
# Compare font counts across all languages at any time:
for lang in telugu tamil malayalam kannada hindi bengali gujarati punjabi odia marathi; do
    n=$(ls fonts/$lang/*.ttf 2>/dev/null | wc -l)
    echo "$lang: $n fonts"
done
```
