# MIST detection + end-to-end pipeline

Scene text detection (DBNet++) and end-to-end detection+recognition code
for the MIST benchmark, used for the IJDAR paper's detection and
end-to-end evaluation sections.

## Contents

- `mist_dbnetpp_config.py`, `mist_dbnetpp_config_extended.py` — MMOCR
  configs for the DBNet++ detector, standard and extended-training
  variants.
- `train_mist_dbnetpp.py`, `train_mist_dbnetpp_extended.py` — training
  entry points for the two configs above.
- `convert_mist_to_mmocr.py`, `repair_manifest.py` — dataset conversion
  and annotation manifest repair utilities to get MIST into MMOCR's
  expected format.
- `run_full_test_inference.py`, `run_full_test_inference_ext.py`,
  `tiled_inference_sample.py` — detector inference on the MIST test
  split (standard and extended checkpoints; tiled inference for
  high-resolution images).
- `run_end_to_end_mist.py`, `run_end_to_end_mist_v2.py` — full detect
  + crop + recognize pipeline.
- `extract_and_route_crops.py`, `build_mist_recognition_gt.py` — crop
  extraction from detector outputs and pseudo-ground-truth construction
  for the recognition stage.
- `deteval_score.py` — DetEval-style detection scoring.
- `score_mist_baselines.py`, `run_mist_baselines.py` — recognition
  baseline scoring/execution over MIST crops.
- `run_track1_clip.py`, `run_vlm_baselines.py`, `run_qwen3vl_baseline.py`,
  `bstd_vlm_infer_unlabeled.py` — CLIP and VLM (including fine-tuned VLM
  and Qwen3-VL) baseline inference used for comparison in the paper.
- `audit_track2_abstention_full.py` — abstention/error analysis over
  recognition predictions.
- `visualize_predictions.py` — visualization of detection/recognition
  predictions for qualitative figures.
- `dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py` — base MMOCR DBNet++
  config this project's configs extend.

## Scope

Code only — no checkpoints, datasets, prediction/log files, or scoring
outputs are included here. See the top-level repository README.
