#!/usr/bin/env bash
#
# train_all_phase1_vlm.sh -- runs phase1_vlm_lora_train.py + phase1_vlm_lora_eval.py
# back to back for all 10 Phase 1 / IndicSTR12 languages, sequentially, on a
# single GPU. Designed to survive being interrupted and re-run: each stage
# writes its own completion marker and is skipped on a re-run once done, and
# training itself resumes from the latest HF Trainer checkpoint if one exists
# (via phase1_vlm_lora_train.py's own --resume/--save_steps machinery).
#
# WHERE TO RUN THIS FROM
#   cd ~/multilingual-str-clip   (this script assumes data/<lang>/... and
#   checkpoints/, results/, logs/ are relative to the current directory --
#   matching how phase1_vlm_lora_train.py's own usage examples are written)
#
# BEFORE THE FULL RUN: SMOKE TEST ONE LANGUAGE FIRST
#   Do NOT launch this script cold. Confirm the environment + a single
#   training step works first, exactly like every other script in this
#   project:
#     python3 phase1_vlm_lora_train.py \
#         --train_tsv data/telugu/train/train.tsv \
#         --image_root data/telugu/train/images \
#         --output_dir checkpoints/telugu_qwen3vl_lora_visiontuned_smoketest \
#         --tune_vision --max_pixels 200704 \
#         --max_train_samples 8 --num_train_epochs 1
#   If that finishes cleanly (check the printed "Trainable params in vision
#   tower" line is > 0, confirming --tune_vision actually took), delete the
#   smoketest output_dir and launch this driver.
#
# TIME ESTIMATE (read before launching)
#   Each language's real train.tsv has ~17,000-20,000 samples (vs. BSTD
#   Kannada's 1,986 real / 33,674 augmented). At batch=1 x grad_accum=8,
#   that is roughly 2,100-2,500 optimizer steps for 1 epoch. BSTD Kannada's
#   --tune_vision run on this same GPU processed steps at ~2.9s/step
#   (4,210 steps in 3h22m), so budget roughly 1.5-2h of TRAINING per
#   language at the default --num_train_epochs 1, plus eval (greedy,
#   1 image at a time, ~5,000 test images per language for most languages)
#   which could add a comparable amount of time again. Realistically this
#   is upwards of a day of continuous GPU time for all 10 languages -- run
#   it under nohup, check in periodically, and expect it to span multiple
#   sessions. It is safe to Ctrl-C and re-run this script at any point;
#   at worst you lose up to --save_steps (300) training steps of progress
#   on whichever language was mid-training.
#
# USAGE
#   nohup bash train_all_phase1_vlm.sh > phase1_vlm_run.log 2>&1 &
#   tail -f phase1_vlm_run.log
#   # or, to check progress at any time without tailing a possibly-stale
#   # buffered log, just look for completion markers:
#   ls checkpoints/*_qwen3vl_lora_visiontuned/.TRAIN_DONE results/.*_EVAL_DONE 2>/dev/null
#
set -u

LANGS=(bengali gujarati hindi kannada malayalam marathi odia punjabi tamil telugu)

mkdir -p checkpoints results logs

for lang in "${LANGS[@]}"; do
    OUT="checkpoints/${lang}_qwen3vl_lora_visiontuned"
    DONE_TRAIN="$OUT/.TRAIN_DONE"
    RESULTS_CSV="results/${lang}_qwen3vl_lora_predictions.csv"
    REPORT="results/${lang}_qwen3vl_lora_report.json"
    DONE_EVAL="results/.${lang}_EVAL_DONE"

    echo ""
    echo "################################################################"
    echo "### $lang -- training"
    echo "################################################################"
    if [ -f "$DONE_TRAIN" ]; then
        echo "  [$lang] training already marked done ($DONE_TRAIN exists) -- skipping."
    else
        RESUME_FLAG=""
        if compgen -G "$OUT/checkpoint-*" > /dev/null; then
            RESUME_FLAG="--resume"
            echo "  [$lang] found existing checkpoint-* under $OUT -- resuming."
        fi
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 phase1_vlm_lora_train.py \
            --train_tsv "data/${lang}/train/train.tsv" \
            --image_root "data/${lang}/train/images" \
            --output_dir "$OUT" \
            --tune_vision --max_pixels 200704 \
            $RESUME_FLAG \
            2>&1 | tee -a "logs/${lang}_train.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "!! [$lang] training FAILED (exit $status) -- stopping driver so you can investigate." >&2
            echo "   Re-run this script after fixing the issue; $lang will resume from its last checkpoint." >&2
            exit 1
        fi
        touch "$DONE_TRAIN"
        echo "  [$lang] training complete."
    fi

    echo ""
    echo "################################################################"
    echo "### $lang -- eval (Phase 1 IndicSTR12 test split)"
    echo "################################################################"
    if [ -f "$DONE_EVAL" ]; then
        echo "  [$lang] eval already marked done ($DONE_EVAL exists) -- skipping."
    else
        python3 phase1_vlm_lora_eval.py \
            --adapter_dir "$OUT" \
            --test_tsv "data/${lang}/test/test.tsv" \
            --image_root "data/${lang}/test/images" \
            --output_csv "$RESULTS_CSV" \
            --output_report "$REPORT" \
            --language "$lang" \
            2>&1 | tee -a "logs/${lang}_eval.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "!! [$lang] eval FAILED (exit $status) -- stopping driver so you can investigate." >&2
            exit 1
        fi
        touch "$DONE_EVAL"
        echo "  [$lang] eval complete -- report at $REPORT"
    fi
done

echo ""
echo "################################################################"
echo "### ALL 10 LANGUAGES COMPLETE -- summary"
echo "################################################################"
for lang in "${LANGS[@]}"; do
    REPORT="results/${lang}_qwen3vl_lora_report.json"
    if [ -f "$REPORT" ]; then
        python3 -c "
import json
r = json.load(open('$REPORT'))
m = r['metrics']
c = r.get('comparison', {})
print(f\"{r['language']:10s}  Qwen3-VL-LoRA WRR={m['WRR']*100:5.2f}%  CRR={m['CRR']*100:5.2f}%\" +
      (f\"   |  PARSeq(pub)={c.get('parseq_published_wrr_pct','?'):5.2f}%  CLIP(this work)={c.get('clip_this_work_wrr_pct','?'):5.2f}%\" if c else ''))
"
    else
        echo "$lang       (no report found)"
    fi
done
