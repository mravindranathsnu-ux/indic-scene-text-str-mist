# Multilingual Indic Scene Text: Recognition + Detection

Code accompanying the IJDAR paper on scene text recognition and detection
across Indian scripts, evaluated on IndicSTR12, BSTD, and MIST.

This repository combines two sub-projects:

- **[`multilingual-str-clip/`](multilingual-str-clip/)** — CLIP-backbone
  scene text *recognition* model (one fine-tuned model per script: Hindi,
  Telugu, Tamil, Malayalam, Kannada, Bengali, Gujarati, Punjabi, Odia,
  English). Includes training/evaluation scripts, vocab and character-list
  definitions per language, synthetic data generation, and lexicon-based
  reranking/correction utilities.
- **[`mist_det_work/`](mist_det_work/)** — scene text *detection* (DBNet++)
  and end-to-end detection+recognition pipeline used for the MIST benchmark
  evaluation, including baselines, DetEval scoring, and VLM baseline
  comparisons (CLIP, fine-tuned VLM).

## Scope of this repository

This repo contains **code only** — no model checkpoints, no datasets, no
generated predictions/logs. See each sub-project's own README and
`requirements.txt` for setup and usage instructions; `multilingual-str-clip`
also has a `scripts/` directory with one shell script per pipeline step
(run from that project's root, e.g. `./scripts/run_telugu.sh`).

Trained checkpoints, the MIST recognition pseudo-ground-truth set, and
full result/scoring artifacts referenced in the paper are available from
the corresponding author on reasonable request.

## Citation

If you use this code, please cite the associated IJDAR paper (citation
details to be added upon publication).
