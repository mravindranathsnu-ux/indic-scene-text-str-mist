# Multilingual Scene Text Recognition via CLIP Transfer Learning

End-to-end project: fine-tune a pretrained **CLIP** vision backbone as an
image encoder, and pair it with a **Transformer decoder** that generates
text character-by-character, for **any of 9 Indian scripts or English**.
This is a *recognition* model — it expects already-cropped word/line
images (from a detector like CRAFT, DBNet, or EAST), not full scene
images with multiple text regions.

This project started as a Telugu-only build (see `telugu-str-clip`, kept
as a separate, untouched reference project) and was generalized from
there — Telugu remains the most thoroughly validated language here, and
is the recommended starting point/template for adding a new language.

**One model = one language.** Train a separate checkpoint per
language/script rather than one combined multilingual model — mixing
scripts in a single small decoder dilutes capacity and complicates the
vocab, which matters a lot given how much data-scarcity affected even a
single language (see the debugging notes throughout this README). The
*code* is shared and generic across all languages; the *trained weights*
are per-language.

## Supported languages

Built-in Unicode-block vocab support (`vocab.py`'s `UNICODE_BLOCKS`) for:

| Language(s) | `--language` value |
|---|---|
| Telugu | `telugu` |
| Hindi, Marathi, Sanskrit, Nepali, ... | `hindi` or `devanagari` |
| Tamil | `tamil` |
| Malayalam | `malayalam` |
| Kannada | `kannada` |
| Bengali | `bengali` |
| Gujarati | `gujarati` |
| Punjabi | `gurmukhi` or `punjabi` |
| Odia | `odia` |
| English | `english` or `latin` |

Adding a new script: add its Unicode block range(s) to `UNICODE_BLOCKS`
in `vocab.py` — everything else in the project is already script-agnostic.

## Why CLIP here, and how

CLIP's text encoder is built for image-text similarity, not per-character
generation, so it isn't used directly. Instead we reuse CLIP's **visual
ViT backbone** as a pretrained feature extractor (transfer learning) and
attach a task-specific sequence decoder on top:

```
image --> CLIP ViT patch embeddings (frozen, last N blocks fine-tuned)
       --> linear projection
       --> Transformer decoder (cross-attends to patch embeddings)
       --> per-step softmax over your language's character vocabulary
```

We deliberately use CLIP's *patch token sequence* rather than its single
pooled embedding, since recognition needs spatial detail (character shapes
and left-to-right order) that pooling would discard.

### Why an English-pretrained CLIP works for any script

CLIP's **text encoder/tokenizer is never used** in this project — only its
**visual ViT backbone**. That backbone learned general-purpose visual
features (edges, strokes, textures, shapes) from natural images, which is
language-agnostic. All the language-specific components — the character
vocabulary, the token embeddings, the decoder, and the output softmax —
are randomly initialized and trained entirely on your target-language
data. So "the CLIP checkpoint was trained on English image-text pairs"
has no direct bearing on whether it can recognize Tamil, Hindi, or any
other script's glyphs; what matters is whether the visual backbone's
low/mid-level features (learned from millions of natural photos, not
language) transfer to reading text strokes, which they generally do.

## Project structure

```
multilingual-str-clip/
├── clip_vit.py       # self-contained CLIP ViT loader — reads .pt checkpoints, no `clip` package
├── vocab.py           # multi-script character vocabulary (UNICODE_BLOCKS registry)
├── convert_char_list.py # convert plain char-list .txt files to Vocab-compatible JSON
├── dataset.py           # TSV-based dataset + collate function
├── split_annotations.py  # split one gt.txt into train.tsv / val.tsv / test.tsv
├── combine_images_labels.py # combine separate images.txt/labels.txt into a TSV
├── generate_synthetic_data.py  # render synthetic word images to supplement real data
├── verify_fonts.py               # check a language's fonts shape/render correctly before using them
├── model.py               # CLIPPatchEncoder (uses clip_vit.py) + Transformer decoder (CLIPSceneTextOCR)
├── train.py                 # two-stage training loop, checkpointing, CER-based model selection
├── evaluate.py                # WRR / CRR (and CER / WER) evaluation on a labeled test set
├── sanity_check.py              # debug script: data checks, preprocessing viz, tiny-batch overfit test
├── check_gpu.py                   # standalone GPU/CUDA environment check
├── check_image_duplicates.py        # detect byte-identical images leaking across train/test splits
├── inference.py                # run a trained checkpoint on new images
├── utils.py                       # CER / WER metrics, device logging
└── requirements.txt
```

## Multi-language project layout

Keep a subfolder per language under each of these, so nothing from one
language overwrites another. **Each split (train/val/test) gets its own
TSV *and* its own images folder** — this project doesn't assume a single
shared image folder across splits, since a pre-split image database
(three separate sets, as many datasets ship) is the common case:

```
multilingual-str-clip/
├── vocab/
│   ├── telugu_vocab.json
│   ├── hindi_vocab.json
│   ├── tamil_vocab.json
│   └── ...
├── fonts/
│   ├── telugu/       (Noto Sans/Serif Telugu, Ramabhadra, Mallanna, ...)
│   ├── hindi/        (Noto Sans/Serif Devanagari, Mukta, ...)
│   ├── tamil/        (Noto Sans/Serif Tamil, Catamaran, Hind Madurai, ...)
│   └── ...           (search fonts.google.com by language — the Noto
│                       family has correct shaping for every supported script)
├── data/
│   ├── telugu/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── train.tsv
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── val.tsv
│   │   └── test/
│   │       ├── images/
│   │       └── test.tsv
│   ├── hindi/
│   │   └── ... (same structure)
│   └── ...
└── checkpoints/
    ├── telugu/{best.pt,last.pt}
    ├── hindi/{best.pt,last.pt}
    └── ...
```

If your image database is already organized this way (a separate image
set per split, as opposed to one shared folder referenced by all three
TSVs), point each script at the matching folder directly —
`--train_image_root`, `--val_image_root`, `--test_image_root` (see
sections 4-6 below). If you *do* have one shared image folder across
splits instead, `--image_root` still works as a single fallback for all
of them, for backward compatibility with a simpler single-folder layout.
```

## 1. Setup

**Prefer running from the terminal with ready-made scripts?** See
`scripts/` — one numbered bash script per pipeline step (`00a`
vocab conversion, `00b` splitting, `01` setup, `02` GPU check, `03` font
verification, `04` synthetic data, `05` duplicate check, `06` sanity
check, `07` train, `08` evaluate, `09` inference), all sourcing a single
`scripts/config.sh` for paths/hyperparameters — edit that one file per
language, then run the numbered scripts in order. `scripts/
ALL_STEPS_REFERENCE.sh` concatenates all of them for reading the whole
pipeline in one place (documentation only — not meant to be executed
top-to-bottom unattended, since steps 03/04/06 need you to review output
before continuing). The manual, step-by-step commands below are exactly
what these scripts run under the hood.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

This project has **no dependency on the `clip` pip package** (or `ftfy`/
`regex`) at all — `clip_vit.py` is a small, self-contained Vision
Transformer implementation that reads the official OpenAI CLIP `.pt`
checkpoint files directly with plain `torch`/`torchvision`. `requirements.txt`
only needs `torch`, `torchvision`, `Pillow`, `tqdm`, `editdistance`, `numpy`
— nothing fetched from GitHub, and no network access at runtime.

### Using your already-downloaded CLIP weights

Point `--clip_ckpt` at your existing file (shared across all languages —
one CLIP backbone checkpoint, many fine-tuned language-specific decoders)
— no other flag is needed, the architecture (patch size, width, depth,
projection dim) is inferred directly from the checkpoint's own tensor
shapes:

```bash
--clip_ckpt /path/to/ViT-B-32.pt
# or
--clip_ckpt /path/to/ViT-B-16.pt
```

Only the **visual tower** is loaded — CLIP's text encoder/tokenizer is
never used, so `ftfy`/`regex` (needed only for CLIP's text tokenizer)
aren't required either.

### Checking GPU availability

Every script (`train.py`, `evaluate.py`, `inference.py`,
`sanity_check.py`) prints which device it's actually using right at
startup, e.g.:
```
[train.py] Using device: cuda:0  (NVIDIA A100-SXM4-40GB, 39.6 GB)
```
or, if no GPU is found:
```
[train.py] Using device: cpu  (no GPU detected/selected — training and inference will be much slower)
```
To check GPU availability independent of any of those scripts, run:
```bash
python check_gpu.py
```
This reports the PyTorch/CUDA versions, every visible GPU (name, memory,
compute capability), actually runs a real matrix multiply on the GPU (not
just a driver query — catches cases where CUDA "is available" per the
driver but something's still misconfigured), and TF32/cuDNN status. If it
reports CPU-only, the most common causes are: no NVIDIA GPU present, the
NVIDIA driver isn't installed (`nvidia-smi` in a terminal should show your
GPU if it is), PyTorch was installed as the CPU-only build (reinstall the
CUDA build from https://pytorch.org/get-started/locally/), or you're in a
container/VM without GPU passthrough enabled.

## 2. Vocabulary

`vocab/` already has ready-to-use vocab files for 13 languages (Assamese,
Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Meitei/Manipuri,
Odia, Punjabi, Tamil, Telugu, Urdu), converted from plain character-list
files via `convert_char_list.py`:
```bash
python convert_char_list.py --input_dir char_lists/ --output_dir vocab/
```
(one character per line in the input `.txt` — the common CRNN/CTC-style
vocab format — converted to the JSON array `Vocab.load()` expects;
duplicates and blank lines are dropped automatically.)

Note on Urdu specifically: it's a right-to-left script. Nothing in this
project's core pipeline assumes left-to-right (label strings are just
Unicode text; direction is a rendering concern, not a storage one), but
if you generate synthetic Urdu data, verify with `verify_fonts.py` that
your fonts/Pillow's raqm+fribidi support render RTL correctly before
trusting the output.

Two ways to get a vocab for a language, in order of recommendation:

**A) Bring your own vocab file.** If you already have a vocab (a flat
JSON array of unique characters, no special tokens — those are added
automatically), just point `--vocab_path` at it:
```bash
--vocab_path vocab/hindi_vocab.json
```
`train.py` always loads an existing `--vocab_path` file first, before
even looking at `--vocab_source`/`--language` — so this is all you need.

**B) Let the project build one.** If `--vocab_path` doesn't exist yet,
`train.py` builds it from `--vocab_source`:
- `unicode_block` (default): the full Unicode block for `--language`
  (e.g. `--language hindi`) plus common punctuation/digits — guarantees
  no "unseen character" cliff at test time, at the cost of some vocab
  entries having little/no training signal if your dataset is small (see
  the class-weighting note in section 6).
- `train_labels`: only the characters actually seen in `--train_ann` — a
  tighter vocab, better fit for a small dataset, but any character
  missing from training can never be predicted correctly.

```bash
python -c "
from vocab import Vocab
v = Vocab.build_from_unicode_block('hindi')
print(len(v), 'vocab entries')
"
```

## 3. Dataset format

Annotation files are TSV: `image_relative_path<TAB>label`, one sample per
line, UTF-8 encoded, regardless of language. `image_root` is the folder
these paths are relative to.

```
images/img_0001.jpg	తెలుగు
images/img_0002.jpg	హైదరాబాద్
```
```
images/img_0001.jpg	नमस्ते
images/img_0002.jpg	भारत
```

Good sources of Indic scene-text data to build from:
- **IIIT-ILST** (Indian Language Scene Text) — Telugu, Hindi, Malayalam subsets.
- **MLT-2017 / MLT-2019** (ICDAR Multi-Lingual Text) — includes Indic scripts.
- **Bharat Scene Text Dataset (BSTD)**.
- Synthetic data (`generate_synthetic_data.py`, section 7) — very useful
  for bootstrapping before real annotated data is available, or for
  supplementing a small real dataset once you've validated the pipeline.

### Already have a pre-split image database (separate train/val/test image sets)?

Skip `split_annotations.py` entirely — it's only for turning one combined
`gt.txt` into three splits. If your database already ships train/val/test
as distinct image folders with their own ground-truth files, just
reformat each split's ground truth into a TSV (`image_path<TAB>label`,
paths relative to that split's own image folder) and place it alongside
that split's images, matching the layout in "Multi-language project
layout" above. Then point `--train_ann`/`--train_image_root`,
`--val_ann`/`--val_image_root`, `--test_ann`/`--test_image_root` at the
three pairs directly (sections 4-6, 9).

**If your ground truth is two separate parallel files** (e.g.
`train_images.txt` listing filenames and `train_labels.txt` listing the
corresponding labels, line i matching line i — a common raw format), use
`combine_images_labels.py` (or `scripts/00c_combine_images_labels.sh` to
do all three splits in one go) instead of reformatting by hand:
```bash
python combine_images_labels.py \
    --images_txt data/telugu/train/train_images.txt \
    --labels_txt data/telugu/train/train_labels.txt \
    --output data/telugu/train/train.tsv \
    --image_prefix "images/"
```
`--image_prefix` prepends a folder to each entry — use it if
`train_images.txt` has bare filenames (`img_0001.jpg`) rather than paths
already relative to that split's `images/` subfolder. It errors clearly
if the two input files have different line counts (the most common
cause of a silent image/label misalignment).

### Splitting a single gt.txt into train / val / test

If you have one full ground-truth file instead of separate TSVs, use
`split_annotations.py`. **Use `--test_ratio` to get a genuine three-way
split** — `val.tsv` and `test.tsv` serve different purposes and should
never be the same file (or, worse, `test.tsv` reused as a symlink/copy of
`val.tsv`):

```bash
python split_annotations.py \
    --gt gt.txt \
    --out_dir data/hindi \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --dedup
```

This writes `train.tsv`, `val.tsv`, and `test.tsv` — verified disjoint
(no image path appears in more than one file) and exhaustive (every
sample lands in exactly one of the three).

- **`val.tsv`**: used every epoch during training — `train.py` saves
  `checkpoints/<language>/best.pt` based on whichever epoch gets the
  lowest CER on this file. Touch it as often as you like while tuning
  hyperparameters.
- **`test.tsv`**: held out completely — never used for training or
  checkpoint selection. Only evaluate on this once you're done tuning,
  and treat that number as the real one.

**Evaluating on `val.tsv` (or reusing it as `test.tsv`) gives an
optimistically biased number** — it's the exact file checkpoint selection
was fit against, not a measure of generalization to unseen data.

Other options:
- `--val_ratio` / `--test_ratio`: fraction held out for each (0.05-0.15 is
  typical for each; omit `--test_ratio` — or leave it at its default 0 —
  for the old two-way train/val split, if you genuinely don't need a
  separate test set yet).
- `--dedup`: drops duplicate `image_path` entries first, so the same image
  never ends up in more than one split.
- `--sep`: change if `gt.txt` isn't TAB-separated. If your file looks like
  `238_1_33_22_148_59.jpg ప్రియా` (filename, a space, then the label), use
  `--sep " "` — this also tolerates extra/variable whitespace between the
  two fields.
- `--seed`: fixed by default (42) so the split is reproducible; change it
  to get a different random split.

## 4. Fine-tune (two-stage transfer learning)

`train.py` runs a two-stage recipe for adapting a pretrained vision
backbone to a new script, with several additions aimed specifically at
closing the "overfits a tiny batch but generalizes poorly" gap and at
long-tail/conjunct characters (present in most Indic scripts):

- **Stage 1 — `--warmup_frozen_epochs`** (default 3): the CLIP backbone is
  fully frozen and used as a pure feature extractor. Only the new decoder
  trains, so the randomly-initialized head converges before touching the
  pretrained weights.
- **Stage 2 — after warmup**: the last `--unfreeze_last_n` transformer
  blocks unfreeze and fine-tune at a much smaller learning rate
  (`--backbone_lr`) than the decoder (`--lr`). The backbone's LR now also
  does its own short **linear warmup** (`--backbone_warmup_steps`) right
  at the moment it unfreezes, instead of jumping straight to full LR —
  hitting a pretrained backbone with a full-strength LR step instantly is
  a common cause of the exact "tiny batch overfits fine, full dataset
  doesn't" pattern, since it can destabilize/forget what the backbone
  already knew partway through training.
- **Cosine LR decay** for both the head and backbone (each with its own
  warmup), down to `--min_lr_frac` of peak LR by the end of training,
  instead of a flat learning rate the whole time.
- **Class-weighted loss** (`--class_weights`, on by default): weights
  each character inversely to its frequency in `--train_ann`, so rare
  long-tail characters (e.g. Telugu/Tamil/Kannada conjuncts, rare
  Devanagari ligatures) get more learning signal instead of being
  drowned out by common vowels/consonants that appear in nearly every
  word. Disable with `--no_class_weights`.
- **Training-time augmentation** (`--augment`, on by default): random
  rotation, perspective warp, blur, and brightness/contrast/saturation
  jitter, applied only to the training split. CLIP's backbone was
  pretrained on clean natural photos, not real-world scene-text crops, so
  this closes some of the domain gap and directly targets overfitting on
  a limited training set. Disable with `--no_augment`.
- **Mixed precision (AMP)** and **gradient accumulation**, for single-GPU
  efficiency — see the dedicated section below.
- **Early stopping** (`--patience`, default 8): stops training if val CER
  hasn't improved for that many consecutive epochs, so you don't have to
  guess `--epochs` up front. Guaranteed not to fire before Stage-2
  warmup genuinely completes (see the debugging notes in section 9) —
  earlier versions of this script had a bug where a longer `--epochs`
  run's proportionally longer warmup could trigger early stopping before
  the model got a fair shot at its peak learning rate.

```bash
python train.py \
    --train_ann data/hindi/train/train.tsv \
    --val_ann data/hindi/val/val.tsv \
    --train_image_root data/hindi/train/images \
    --val_image_root data/hindi/val/images \
    --clip_ckpt ./weights/ViT-B-32.pt \
    --language hindi \
    --vocab_path vocab/hindi_vocab.json \
    --out_dir checkpoints/hindi \
    --batch_size 32 \
    --epochs 40 \
    --warmup_frozen_epochs 3 \
    --unfreeze_last_n 2 \
    --lr 3e-4 \
    --backbone_lr 3e-5
```

(`--language` only matters if `--vocab_path` doesn't already exist and
`--vocab_source unicode_block` — if you're bringing your own vocab file,
as recommended in section 2, `--language` has no effect.)

Key transfer-learning knobs:
- `--clip_ckpt`: path to your local checkpoint (`ViT-B-32.pt`, `ViT-B-16.pt`,
  etc). `ViT-B-16` has finer patches (better for small/dense conjunct
  glyphs) but is slower; `ViT-B-32` is faster and a good default to start.
- `--warmup_frozen_epochs`: increase this (e.g. 5-10) if you have very
  little labeled data for this language.
- `--unfreeze_last_n`: how deep into the backbone fine-tuning reaches.
  Start at 2; raise to 4-6 only with several thousand+ labeled samples.
- `--backbone_lr` vs `--lr`: keep `backbone_lr` at least 5-10x smaller
  than `lr`.
- `--head_warmup_steps` / `--backbone_warmup_steps`: **default to 10% of
  total training steps** (computed from your actual dataset size/batch
  size/epochs, printed at the start of training as e.g. `Total steps:
  1400  Head warmup: 140 steps`). Don't hardcode a large fixed number
  here (e.g. 500) without checking it against your total step count —
  on a small dataset that can silently consume most/all of training
  with the LR still ramping up, showing up as `val_CER` stuck near its
  worst value with no improvement for many epochs.

Training prints train/val loss, validation **CER**, and the current
backbone/head learning rates each epoch, and saves `best.pt`
(lowest CER) and `last.pt` under `--out_dir`.

### Single-GPU training efficiency

- `--amp` (on by default on CUDA, no-op on CPU): mixed-precision
  training — meaningfully faster and lower memory on a single GPU, with
  gradient scaling handled automatically. `evaluate.py` and
  `inference.py` also use autocast during inference (their own `--amp`
  flag, on by default).
- `--grad_accum_steps`: accumulate gradients over this many batches
  before each optimizer step, to simulate a larger effective batch size
  when your GPU can't fit a large `--batch_size` directly (e.g.
  `--batch_size 16 --grad_accum_steps 2` behaves like batch size 32).
- `--num_workers`: DataLoader worker count — lower this if you're on a
  machine with few CPU cores alongside the single GPU.
- **TF32 + cuDNN autotuning**: enabled automatically whenever `--device`
  resolves to CUDA — free speedup, no flag needed.
- `--compile` (off by default): wraps the model with `torch.compile()`
  (PyTorch 2.0+, CUDA only) for extra speedup, at the cost of a one-time
  compilation delay at the start of training.
- **Batched inference**: `inference.py --image_dir` processes images
  in batches (`--batch_size`, default 16) instead of one at a time.

## 5. Inference

```bash
python inference.py --ckpt checkpoints/hindi/best.pt --clip_ckpt ./weights/ViT-B-32.pt --image_dir samples/
```

No language flag needed — the vocab is embedded in the checkpoint
(`vocab_chars`) and reconstructed automatically. Prints
`filename<TAB>predicted_text` for every image, using greedy decoding by
default. Add `--decode beam --beam_size 5` for beam search, which usually
gives a small accuracy bump over greedy (it can recover from a
locally-plausible-but-wrong character choice — useful for visually
similar characters/conjuncts in any script) at the cost of somewhat
slower inference. `--length_penalty` (default 0.7) controls how strongly
beam search favors longer completions — raise it (e.g. 1.0-1.5) if you
see predictions systematically truncated on longer words.

## 6. Evaluate: WRR / CRR (overall accuracy)

`evaluate.py` runs a trained checkpoint over a labeled test set and reports
the two headline scene-text-recognition metrics:

- **Word Recognition Rate (WRR)** — the fraction of images whose predicted
  text is an *exact* match to the ground truth. This is the standard
  "word accuracy" figure reported in STR papers/benchmarks — no partial
  credit, each sample is either fully right or fully wrong.
- **Character Recognition Rate (CRR)** — `1 - CER`, i.e. character-level
  accuracy based on edit distance, aggregated over the whole test set.

It also prints the underlying CER/WER (error rates) for reference.

```bash
python evaluate.py \
    --ckpt checkpoints/hindi/best.pt \
    --clip_ckpt ./weights/ViT-B-32.pt \
    --test_ann data/hindi/test/test.tsv \
    --test_image_root data/hindi/test/images \
    --decode beam --beam_size 5 \
    --out_csv results/hindi_test_predictions.csv
```

Example output:
```
==============================================
 SCENE TEXT RECOGNITION — EVALUATION REPORT
==============================================
 Samples evaluated               : 1240
 Word Recognition Rate  (WRR)    :  78.31 %
 Character Recognition Rate (CRR):  93.42 %
----------------------------------------------
 Word Error Rate       (WER)     :  21.69 %
 Character Error Rate  (CER)     :   6.58 %
==============================================
```

`--out_csv` (optional) writes every sample's `image_path, ground_truth,
prediction, exact_match` to a CSV — useful for manually scanning
mispredictions to spot patterns.

## 7. Fonts and synthetic data generation

If real hyperparameter tuning (augmentation, class weighting, backbone
unfreezing, LR schedule) has plateaued — `train_loss` flattening at a
similar level across several different configurations is the tell — the
dataset itself is likely the limiting factor. `generate_synthetic_data.py`
renders additional word-crop images to supplement real data, for any
script, given real font files for it.

### Verify fonts before generating at scale

```bash
python verify_fonts.py --fonts_dir fonts/hindi --language hindi
```
Renders a sample word (with conjuncts/matras, chosen per-language to
stress-test shaping) with every font in the directory, and reports which
ones fail to load. **Open the output PNGs and visually confirm correct
shaping** before trusting a font — a font/Pillow combination that
silently mis-shapes complex scripts would poison your synthetic data with
garbled labels. Requires Pillow's `raqm`/harfbuzz support:
```bash
python -c "from PIL import features; print(features.check('raqm'))"
```
If that prints `False`, `pip install --upgrade --force-reinstall Pillow`
usually pulls a wheel with `raqm` bundled.

### Generate

```bash
python generate_synthetic_data.py \
    --train_ann data/hindi/train/train.tsv \
    --fonts_dir fonts/hindi \
    --out_dir data/hindi/train/synthetic \
    --samples_per_word 6 \
    --length_boost 0.8 \
    --exclude_ann data/hindi/val/val.tsv data/hindi/test/test.tsv
```

**Always pass `--exclude_ann <val.tsv> <test.tsv>`.** Without it, any word
that appears in `train.tsv` gets synthetic-augmented even if that *same
word* (as label text, not necessarily the same image) also appears in
your val/test set — very likely for repeated signage vocabulary in any
language. That gives the model heavy memorized exposure to that exact
text before it ever needs to generalize to it, quietly inflating
evaluation numbers without showing up as an image-path overlap. This is
a real leakage vector this project ran into directly during Telugu
development — omitting `--exclude_ann` produced a checkpoint that scored
100% on every single test sample, which was not genuine generalization.

Two things it does, both usable together:
- **Style augmentation** (always on): re-renders every word already in
  your real `train.tsv` in multiple fonts/sizes/colors/rotations — more
  visual diversity per word without needing new vocabulary.
- **Vocabulary expansion** (`--corpus_file extra_words.txt`, optional): a
  plain text file of additional real words in your target script, one
  per line (UTF-8) — expands word/character-context diversity beyond
  your labeled set.

`--length_boost` (0-1+, default 0) oversamples longer words — real
training data usually skews toward shorter words, so without this the
decoder gets proportionally less "keep going" signal at deeper sequence
positions, which shows up as systematic premature-EOS truncation on
longer words specifically (check with a length-bucketed WRR/CRR
breakdown on your eval predictions — see section 9).

Combine with real data by concatenating TSVs:
```bash
cat data/hindi/train/train.tsv data/hindi/train/synthetic/synthetic.tsv > data/hindi/train/train_augmented.tsv
```
Since `generate_synthetic_data.py`'s `--out_dir` was placed inside the
same `train/` folder here, synthetic image paths already resolve under
`--train_image_root data/hindi/train/images` automatically as long as you
either point `--out_dir` at `data/hindi/train/images/synthetic` (so the
`images/` prefix in `synthetic.tsv` lines up), or symlink the synthetic
`images/` output into your real train image folder — the simplest way to
guarantee a single `--train_image_root` resolves every path in
`train_augmented.tsv`, real and synthetic alike.

## 8. Preprocessing: LetterboxResize

`clip_vit.py` uses `LetterboxResize`, not CLIP's default preprocessing
(`Resize(shorter side)` → `CenterCrop(square)`). That default is correct
for natural photos but **destroys wide/short OCR word crops** — e.g. a
150×40 crop gets scaled to ~840×224, then center-cropped down to 224×224,
keeping only the middle ~27% of the width. That alone can produce
near-0% Word Recognition Rate while Character Recognition Rate stays
just barely above chance.

Since this project's images are already tightly cropped to the text,
`LetterboxResize` scales the whole image to fit inside the square input
frame with its aspect ratio fully preserved, then pads with neutral gray.
No cropping, no stretching — every character stays intact and in its
true proportions, which matters across scripts since vowel signs/matras
in most Indic scripts can become hard to distinguish if vertically
stretched.

## 9. Debugging poor accuracy

If WRR/CRR come back low — **especially if a tiny-batch overfit test
passes at 100% but the full test set doesn't** — run `sanity_check.py`
before assuming it's purely a training-length/data-size issue:

```bash
python sanity_check.py \
    --train_ann data/hindi/train/train.tsv \
    --train_image_root data/hindi/train/images \
    --clip_ckpt ./weights/ViT-B-32.pt \
    --test_ann data/hindi/test/test.tsv \
    --vocab_path vocab/hindi_vocab.json \
    --overfit_unfreeze_last_n 2 --overfit_backbone_lr 3e-5
```

It runs four checks in order of how likely each is to explain a bad
result:

1. **Data sanity** — parses your TSV, reports image aspect ratios (flags
   wide/short crops), and checks vocabulary coverage against your
   *actual* `--vocab_path` file (not a silently-rebuilt one) for both
   train and test sets.
2. **Preprocessing visualization** — saves "what the model actually sees"
   PNGs to `sanity_preproc_samples/`. Open a few and check the whole word
   is visible and legible.
3. **Tiny-batch overfit test, backbone frozen** (Stage 1 equivalent) —
   trains on ~8 samples with the backbone frozen and checks the model can
   memorize them almost perfectly (using the best checkpoint seen during
   the test, not just the final step — constant-LR training past
   convergence on a tiny batch commonly destabilizes late, which is a
   training-test artifact, not necessarily a pipeline bug). If this
   fails even at its best point, the core pipeline (data pairing,
   causal/padding mask, loss, decoding) has a bug independent of the
   two-stage schedule.
4. **Tiny-batch overfit test, backbone unfrozen** (Stage 2 equivalent) —
   the same test, but with the last `--overfit_unfreeze_last_n` CLIP
   blocks unfrozen and the same discriminative LR setup as `train.py`.
   **This is the check that catches "tiny batch = 100%, full test =
   9%"**: if Stage 1 passes but Stage 2 fails, the bug is specific to
   backbone fine-tuning — most likely `--backbone_lr` too high,
   insufficient LR warmup at the unfreeze point, or `--unfreeze_last_n`
   too large for your dataset size.

The script ends with an overall verdict combining all four results.

### If everything checks out but accuracy is still surprisingly good (or bad)

Two things worth ruling out explicitly, both encountered during this
project's development:
- **Leakage via synthetic data**: always use `--exclude_ann` (section 7).
- **Duplicate images under different filenames**: real scene-text
  datasets can have overlapping/duplicate bounding-box annotations that
  produce the same physical crop under two different filenames — path-
  string-based train/val/test splitting won't catch this.
  `check_image_duplicates.py` hashes actual image bytes to check:
  ```bash
  python check_image_duplicates.py \
      --train_image_root data/hindi/train/images \
      --test_image_root data/hindi/test/images \
      --train_ann data/hindi/train/train_augmented.tsv \
      --test_ann data/hindi/test/test.tsv
  ```

### Tips for better accuracy, by lever

These are implemented already (see flags above) — this is about how to
tune them:
- **Augmentation**: often the single biggest lever for closing a
  "memorizes training data but doesn't generalize" gap on a small
  dataset — verified directly during Telugu development (disabling it
  roughly halved WRR).
- **Long-tail/conjunct characters**: `--class_weights` (on by default)
  helps, but matters most in combination with a `train_labels`-derived
  vocab on a small dataset — a `unicode_block` vocab on a small dataset
  means many entries see near-zero training examples regardless of
  weighting.
- **Beam search**: a small accuracy bump for free once training itself
  is solid — don't use it to paper over a training-side issue.
- **LR warmup at the Stage-2 unfreeze point**: if `sanity_check.py`'s
  Stage-2 test shows loss spiking, raise `--backbone_warmup_steps`, or
  lower `--backbone_lr`, or reduce `--unfreeze_last_n`.
- **More epochs / early stopping**: prefer setting `--epochs` generously
  high (60-100+) and let `--patience` decide, rather than under-training
  by guessing a low epoch count.
- **Dataset size**: if none of the above closes the gap, it's most
  likely simply not enough labeled data for this language. See section 7
  for synthetic data generation, particularly `--length_boost` and a
  `--corpus_file` for genuinely new vocabulary (not just style variation
  of words the model already knows).
