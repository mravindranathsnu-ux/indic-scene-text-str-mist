#!/bin/bash
# Runs every step for gujarati, start to finish: setup through the sanity
# check, then asks whether to continue straight into training. Thin
# wrapper around run_pretrain_pipeline.sh — see that file for details.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_pretrain_pipeline.sh" gujarati
