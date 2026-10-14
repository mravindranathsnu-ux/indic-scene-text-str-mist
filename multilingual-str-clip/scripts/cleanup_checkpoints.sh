#!/bin/bash
# cleanup_checkpoints.sh — frees disk space by removing intermediate
# training artifacts (checkpoint_epoch_*.pt periodic snapshots, last.pt,
# resume_state.pt), keeping only best.pt.
#
# SAFETY RULE: never deletes anything until a verified, up-to-date backup
# of best.pt exists under a rolling "current_best" label. If a
# "current_best" backup already exists and matches the live best.pt
# (checked via md5sum, not just presence), it's left alone and cleanup
# proceeds directly. If it's missing or stale (best.pt has changed since
# it was taken), a fresh one is made first. This is separate from any
# named milestone backups (e.g. "unfreeze8_best", "final_locked") made
# via checkpoint_backup.sh -- those are untouched by this script.
#
# Usage:
#   ./scripts/cleanup_checkpoints.sh <language>
#   ./scripts/cleanup_checkpoints.sh all
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

VALID_LANGUAGES="bengali gujarati hindi kannada malayalam marathi odia punjabi tamil telugu"

LANG_ARG="${1:-}"
if [ -z "$LANG_ARG" ]; then
    echo "Usage: $0 <language>"
    echo "  e.g. $0 telugu"
    echo "  or:  $0 all"
    exit 1
fi

if [ "$LANG_ARG" != "all" ]; then
    case " $VALID_LANGUAGES " in
        *" $LANG_ARG "*) ;;
        *)
            echo "!! Unknown language: '$LANG_ARG'"
            echo "!! Supported: $VALID_LANGUAGES"
            echo "!! Or use: $0 all"
            exit 1
            ;;
    esac
fi

human_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1} bytes"
}

cleanup_one() {
    local lang="$1"
    local ckpt_dir="$PROJECT_ROOT/checkpoints/$lang"
    local backup_dir="$PROJECT_ROOT/checkpoints/${lang}_backups/current_best"
    local results_dir="$PROJECT_ROOT/results"

    echo "=== $lang ==="

    if [ ! -f "$ckpt_dir/best.pt" ]; then
        echo "  [SKIP] no checkpoints/$lang/best.pt -- nothing to clean or back up."
        return
    fi

    # Step 1: ensure a verified, up-to-date "current_best" backup exists.
    local need_backup=0
    if [ ! -f "$backup_dir/best.pt" ]; then
        echo "  No current_best backup found -- creating one."
        need_backup=1
    else
        live_sum=$(md5sum "$ckpt_dir/best.pt" | awk '{print $1}')
        backup_sum=$(md5sum "$backup_dir/best.pt" | awk '{print $1}')
        if [ "$live_sum" != "$backup_sum" ]; then
            echo "  current_best backup is STALE (best.pt has changed since it was taken) -- refreshing."
            need_backup=1
        else
            echo "  [OK] current_best backup is verified up to date (checksums match)."
        fi
    fi

    if [ $need_backup -eq 1 ]; then
        mkdir -p "$backup_dir"
        cp "$ckpt_dir/best.pt" "$backup_dir/best.pt"
        for f in training_summary.txt training_summary.json; do
            [ -f "$ckpt_dir/$f" ] && cp "$ckpt_dir/$f" "$backup_dir/$f"
        done
        for f in "${lang}_test_report.txt" "${lang}_test_report.json" "${lang}_test_predictions.csv"; do
            [ -f "$results_dir/$f" ] && cp "$results_dir/$f" "$backup_dir/$f"
        done
        # Verify the copy actually succeeded before proceeding to delete anything.
        new_sum=$(md5sum "$backup_dir/best.pt" | awk '{print $1}')
        live_sum=$(md5sum "$ckpt_dir/best.pt" | awk '{print $1}')
        if [ "$new_sum" != "$live_sum" ]; then
            echo "  !! Backup verification FAILED after copy -- refusing to delete anything for $lang."
            return
        fi
        echo "  [OK] Backup created and verified."
    fi

    # Step 2: only now, with a verified backup in place, remove intermediates.
    local freed=0
    local count=0
    for f in "$ckpt_dir"/checkpoint_epoch_*.pt "$ckpt_dir/last.pt" "$ckpt_dir/resume_state.pt"; do
        if [ -f "$f" ]; then
            sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            freed=$((freed + sz))
            count=$((count + 1))
            rm -f "$f"
        fi
    done

    if [ $count -eq 0 ]; then
        echo "  Nothing to clean (no intermediate files present)."
    else
        echo "  Removed $count intermediate file(s), freed $(human_size $freed)."
    fi
    echo ""
}

TOTAL_START=$(date +%s)

if [ "$LANG_ARG" = "all" ]; then
    for lang in $VALID_LANGUAGES; do
        cleanup_one "$lang"
    done
else
    cleanup_one "$LANG_ARG"
fi

echo "############################################################"
echo "# Cleanup complete."
echo "############################################################"
