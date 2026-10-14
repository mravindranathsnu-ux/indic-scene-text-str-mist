#!/bin/bash
# Runs every pre-training step for tamil. Thin wrapper around
# run_pretrain_pipeline.sh — see that file for what actually happens.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_pretrain_pipeline.sh" tamil
