#!/bin/bash
# run_unattended.sh — runs any command from this project completely
# unattended, auto-answering every [y/N] confirmation and "press Enter to
# continue" pause along the way -- for overnight runs where no one is
# present to provide input.
#
# Rather than individually tracking down and adding a separate bypass
# flag for every single prompt across every script (preflight checklist,
# checkpoint-clearing, font/synthetic-data review pauses, the
# proceed-to-train prompt, train.py's OWN confirmation nested inside every
# 07_train.sh call -- including every extension round if wrapping
# auto_extend_training.sh -- and evaluate.py's confirmation), this feeds
# an infinite stream of "y" into stdin, which correctly satisfies both
# [y/N] prompts and bare "press Enter" pauses uniformly, cascading through
# every nested child process the wrapped command spawns.
#
# Usage:
#   ./run_unattended.sh ./scripts/run_telugu.sh
#   ./run_unattended.sh ./scripts/auto_extend_training.sh
#   ./run_unattended.sh env FORCE_REGEN=1 ./scripts/run_telugu.sh
#
# IMPORTANT: this skips every human-review checkpoint too (font contact
# sheet review, synthetic image spot-check, sanity check verdict review)
# -- appropriate for a language you've already validated once and are
# now re-running (e.g. round 2 with settings you've already reviewed),
# NOT recommended for a language's very first run where those reviews
# still matter. Also sets TRAIN_YES=1 explicitly as a second, more
# specific safeguard for train.py's confirmation, on top of the piped
# "y" stream.
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 ./scripts/run_telugu.sh"
    echo "Example: $0 ./scripts/auto_extend_training.sh"
    exit 1
fi

export TRAIN_YES=1

echo "############################################################"
echo "# UNATTENDED MODE: $* "
echo "# Every prompt will be auto-answered. Human-review checkpoints"
echo "# (font sheets, synthetic spot-checks, sanity verdicts) are"
echo "# skipped, not verified -- appropriate for a re-run of a"
echo "# language you've already validated, not a first run."
echo "############################################################"
echo ""

yes | "$@"
STATUS=$?

echo ""
echo "############################################################"
echo "# Unattended run finished. Exit code: $STATUS"
echo "############################################################"
exit "$STATUS"
