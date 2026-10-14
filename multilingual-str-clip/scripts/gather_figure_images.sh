#!/bin/bash
# gather_figure_images.sh
# Copies the specific example images selected for the paper's qualitative
# results figure from each language's test/images/ folder into one
# consolidated directory, prefixed by language to avoid filename collisions.
#
# Run this from the project root (~/multilingual-str-clip).
set -e

OUT_DIR="paper_figure_images"
mkdir -p "$OUT_DIR"

# language:source_dir pairs -- adjust source_dir if any language's test
# images live somewhere other than the standard data/<lang>/test/images path.
declare -A SRC_DIR=(
  [telugu]="data/telugu/test/images"
  [tamil]="data/tamil/test/images"
  [kannada]="data/kannada/test/images"
  [malayalam]="data/malayalam/test/images"
  [marathi]="data/marathi/test/images"
  [bengali]="data/bengali/test/images"
  [punjabi]="data/punjabi/test/images"
  [hindi]="data/hindi/test/images"
  [odia]="data/odia/test/images"
  [gujarati]="data/gujarati/test/images"
)

# language:filename pairs, one line per selected image (correct + near-miss)
SELECTIONS="
telugu:07122022_L_GH042769_image_000118_28.png
telugu:07122022_R_GH032759_image_000010_14.png
telugu:23082022_R_GH031375_image_000106_8.png
tamil:09012023_L_GH042787_image_000132_17.png
tamil:09012023_L_GH052785_image_000182_47.png
tamil:09012023_L_GH012785_image_000036_17.png
kannada:15122022_L_GH022773_image_000049_64.png
kannada:30102023_L_GH028942_image_000224_54.png
kannada:30102023_L_GH018943_image_000131_69.png
malayalam:11012023_L_GH012790_image_000003_2.png
malayalam:07102023_L_GH011753_image_000132_23.png
malayalam:11012023_R_GH032801_image_000068_3_7.png
marathi:16122022_L_GH032780_image_000070_13.png
marathi:16122022_R_GH012777_image_000071_8.png
marathi:16122022_L_GH012781_image_000207_28.png
bengali:31052023_L_GH010006_image_000111_8.png
bengali:31052023_L_GH032822_image_000027_4.png
bengali:31052023_L_GH012822_image_000166_8.png
punjabi:16042023_R_GH012848_image_000266_3_21.png
punjabi:16042023_L_GH012847_image_000015_43.png
punjabi:16042023_R_GH032853_image_000326_3_19.png
hindi:13042023_R_GH032830_image_000023_2.png
hindi:13042023_L_GH022838_image_000087_9.png
hindi:13042023_L_GH042839_image_000172_61.png
odia:11032023_R_GH022816_image_000098_1.png
odia:13032023_R_GH032821_image_000206_11.png
odia:13032023_R_GH032824_image_000015_3.png
gujarati:30_15_200_775_311_775_311_831_200_831.jpeg
gujarati:241_25_1130_674_1309_677_1309_762_1130_760.jpeg
gujarati:151_5_616_214_663_214_663_235_616_235.jpeg
"

found=0
missing=0

while IFS=':' read -r lang fname; do
    [ -z "$lang" ] && continue
    src="${SRC_DIR[$lang]}/$fname"
    dest="$OUT_DIR/${lang}_${fname}"
    if [ -f "$src" ]; then
        cp "$src" "$dest"
        found=$((found + 1))
    else
        echo "  !! MISSING: $src"
        missing=$((missing + 1))
    fi
done <<< "$SELECTIONS"

echo ""
echo "############################################################"
echo "# Copied $found image(s) to $OUT_DIR/"
if [ $missing -gt 0 ]; then
    echo "# $missing file(s) not found -- see !! lines above."
    echo "# Check SRC_DIR paths at the top of this script if a whole"
    echo "# language's files are missing, e.g. Gujarati's may live under"
    echo "# a different folder if your local layout differs."
fi
echo "############################################################"
echo ""
echo "Next: zip and share the result:"
echo "  zip -r paper_figure_images.zip $OUT_DIR/"
