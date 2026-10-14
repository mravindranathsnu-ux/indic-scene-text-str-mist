#!/bin/bash
# checkpoint_backup.sh — backup, restore, and list labeled checkpoint
# snapshots for any language, so a risky experiment can always be rolled
# back to the last known-good result without re-running training.
#
# Backs up: best.pt, training_summary.txt/.json, and the matching results
# files (test_report.txt/.json, test_predictions.csv) together, since a
# checkpoint without its documented result is much less useful.
#
# Usage:
#   ./scripts/checkpoint_backup.sh backup  <language> <label>
#   ./scripts/checkpoint_backup.sh restore <language> <label>
#   ./scripts/checkpoint_backup.sh list    <language>
#
# Examples:
#   ./scripts/checkpoint_backup.sh backup  telugu unfreeze8_best
#   ./scripts/checkpoint_backup.sh restore telugu unfreeze8_best
#   ./scripts/checkpoint_backup.sh list    telugu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

usage() {
    echo "Usage:"
    echo "  $0 backup  <language> <label>"
    echo "  $0 restore <language> <label>"
    echo "  $0 list    <language>"
    exit 1
}

ACTION="${1:-}"
LANG_ARG="${2:-}"
LABEL="${3:-}"

if [ -z "$ACTION" ] || [ -z "$LANG_ARG" ]; then
    usage
fi
if [ "$ACTION" != "list" ] && [ -z "$LABEL" ]; then
    usage
fi

BACKUP_ROOT="$PROJECT_ROOT/checkpoints/${LANG_ARG}_backups"
CKPT_DIR="$PROJECT_ROOT/checkpoints/$LANG_ARG"
RESULTS_DIR="$PROJECT_ROOT/results"

case "$ACTION" in

backup)
    BACKUP_DIR="$BACKUP_ROOT/$LABEL"
    if [ -d "$BACKUP_DIR" ]; then
        echo "!! A backup already exists at $BACKUP_DIR"
        echo "!! Choose a different label, or remove it first if you really want to overwrite it:"
        echo "!!   rm -rf \"$BACKUP_DIR\""
        exit 1
    fi

    if [ ! -f "$CKPT_DIR/best.pt" ]; then
        echo "!! No checkpoint found at $CKPT_DIR/best.pt -- nothing to back up."
        exit 1
    fi

    mkdir -p "$BACKUP_DIR"
    echo "Backing up $LANG_ARG checkpoint as '$LABEL'..."

    cp "$CKPT_DIR/best.pt" "$BACKUP_DIR/" && echo "  [OK] best.pt"
    for f in training_summary.txt training_summary.json; do
        if [ -f "$CKPT_DIR/$f" ]; then
            cp "$CKPT_DIR/$f" "$BACKUP_DIR/" && echo "  [OK] $f"
        else
            echo "  [SKIP] $f (not found)"
        fi
    done
    for f in "${LANG_ARG}_test_report.txt" "${LANG_ARG}_test_report.json" "${LANG_ARG}_test_predictions.csv"; do
        if [ -f "$RESULTS_DIR/$f" ]; then
            cp "$RESULTS_DIR/$f" "$BACKUP_DIR/" && echo "  [OK] results/$f"
        else
            echo "  [SKIP] results/$f (not found)"
        fi
    done

    echo ""
    echo "Backed up to: $BACKUP_DIR"
    if [ -f "$BACKUP_DIR/${LANG_ARG}_test_report.txt" ]; then
        echo ""
        grep -E "WRR|CRR" "$BACKUP_DIR/${LANG_ARG}_test_report.txt" | sed 's/^/  /'
    fi
    ;;

restore)
    BACKUP_DIR="$BACKUP_ROOT/$LABEL"
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "!! No backup found at $BACKUP_DIR"
        echo "!! Run '$0 list $LANG_ARG' to see available backups."
        exit 1
    fi
    if [ ! -f "$BACKUP_DIR/best.pt" ]; then
        echo "!! $BACKUP_DIR exists but has no best.pt -- refusing to restore an incomplete backup."
        exit 1
    fi

    echo "Restoring $LANG_ARG checkpoint from backup '$LABEL'..."
    mkdir -p "$CKPT_DIR"
    cp "$BACKUP_DIR/best.pt" "$CKPT_DIR/best.pt" && echo "  [OK] best.pt"
    for f in training_summary.txt training_summary.json; do
        if [ -f "$BACKUP_DIR/$f" ]; then
            cp "$BACKUP_DIR/$f" "$CKPT_DIR/$f" && echo "  [OK] $f"
        fi
    done
    for f in "${LANG_ARG}_test_report.txt" "${LANG_ARG}_test_report.json" "${LANG_ARG}_test_predictions.csv"; do
        if [ -f "$BACKUP_DIR/$f" ]; then
            cp "$BACKUP_DIR/$f" "$RESULTS_DIR/$f" && echo "  [OK] results/$f"
        fi
    done

    echo ""
    echo "Restored. $CKPT_DIR/best.pt now matches backup '$LABEL'."
    echo "NOTE: this does NOT touch checkpoints/$LANG_ARG/last.pt, checkpoint_epoch_*.pt,"
    echo "or resume_state.pt -- if any of those still exist from a newer, unwanted run,"
    echo "remove them too before re-running the pipeline, or it may try to resume from them."
    ;;

list)
    if [ ! -d "$BACKUP_ROOT" ] || [ -z "$(ls -A "$BACKUP_ROOT" 2>/dev/null)" ]; then
        echo "No backups found for $LANG_ARG (looked in $BACKUP_ROOT)."
        exit 0
    fi
    echo "Backups available for $LANG_ARG:"
    echo ""
    for d in "$BACKUP_ROOT"/*/; do
        label=$(basename "$d")
        report="$d/${LANG_ARG}_test_report.txt"
        echo "  $label"
        if [ -f "$report" ]; then
            grep -E "WRR|CRR" "$report" | sed 's/^/    /'
        else
            echo "    (no results report saved with this backup)"
        fi
        echo ""
    done
    ;;

*)
    usage
    ;;
esac
