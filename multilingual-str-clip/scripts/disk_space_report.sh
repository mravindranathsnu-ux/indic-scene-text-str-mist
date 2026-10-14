#!/bin/bash
# disk_space_report.sh — runs a full disk-space diagnostic and saves it to a
# single report file you can share back. Read-only: this script never
# deletes or modifies anything, it only inspects and reports.
#
# Usage:
#   ./disk_space_report.sh
#   (writes disk_space_report_<timestamp>.txt in the current directory, and
#    also prints everything to the terminal as it runs)

REPORT="disk_space_report_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "============================================================"
    echo " DISK SPACE REPORT"
    echo " Generated: $(date)"
    echo " Host: $(hostname 2>/dev/null)"
    echo "============================================================"

    echo ""
    echo "------------------------------------------------------------"
    echo " 1. Overall disk space (home directory's filesystem)"
    echo "------------------------------------------------------------"
    df -h "$HOME" 2>/dev/null

    echo ""
    echo "------------------------------------------------------------"
    echo " 2. Biggest top-level folders in your home directory"
    echo "------------------------------------------------------------"
    du -h --max-depth=1 "$HOME" 2>/dev/null | sort -rh | head -20

    echo ""
    echo "------------------------------------------------------------"
    echo " 3. Cache directories (safe to clear via pip cache purge /"
    echo "    conda clean --all -- see the guidance that comes with this report)"
    echo "------------------------------------------------------------"
    if [ -d "$HOME/.cache" ]; then
        du -sh "$HOME"/.cache/* 2>/dev/null | sort -rh | head -15
    else
        echo "  (no ~/.cache directory found)"
    fi

    echo ""
    for d in "$HOME/anaconda3/pkgs" "$HOME/miniconda3/pkgs"; do
        if [ -d "$d" ]; then
            echo "  $d: $(du -sh "$d" 2>/dev/null | cut -f1)"
        fi
    done

    PIP_CACHE_DIR=$(python3 -m pip cache dir 2>/dev/null)
    if [ -n "$PIP_CACHE_DIR" ] && [ -d "$PIP_CACHE_DIR" ]; then
        echo "  pip cache ($PIP_CACHE_DIR): $(du -sh "$PIP_CACHE_DIR" 2>/dev/null | cut -f1)"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " 4. Trash (already 'deleted' but not yet freed)"
    echo "------------------------------------------------------------"
    if [ -d "$HOME/.local/share/Trash" ]; then
        du -sh "$HOME/.local/share/Trash" 2>/dev/null
    else
        echo "  (no Trash directory found)"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " 5. Downloads folder contents, largest first"
    echo "------------------------------------------------------------"
    if [ -d "$HOME/Downloads" ]; then
        du -sh "$HOME"/Downloads/* 2>/dev/null | sort -rh | head -20
    else
        echo "  (no ~/Downloads directory found)"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " 6. multilingual-str-clip project footprint, by top-level folder"
    echo "------------------------------------------------------------"
    PROJECT_DIR=""
    for candidate in "$HOME/multilingual-str-clip" "$(pwd)"; do
        if [ -d "$candidate/data" ] && [ -d "$candidate/scripts" ]; then
            PROJECT_DIR="$candidate"
            break
        fi
    done
    if [ -n "$PROJECT_DIR" ]; then
        echo "  Project found at: $PROJECT_DIR"
        du -h --max-depth=1 "$PROJECT_DIR" 2>/dev/null | sort -rh
        echo ""
        echo "  Per-language breakdown (data/ subfolder):"
        du -h --max-depth=1 "$PROJECT_DIR/data" 2>/dev/null | sort -rh
        echo ""
        echo "  Per-language breakdown (checkpoints/ subfolder):"
        du -h --max-depth=1 "$PROJECT_DIR/checkpoints" 2>/dev/null | sort -rh
        echo ""
        echo "  incoming/ (raw archives + extracted staging):"
        du -h --max-depth=1 "$PROJECT_DIR/incoming" 2>/dev/null | sort -rh
    else
        echo "  (multilingual-str-clip project directory not found automatically —"
        echo "   run this script from inside it, or edit PROJECT_DIR detection above)"
    fi

    echo ""
    echo "============================================================"
    echo " END OF REPORT"
    echo "============================================================"

} | tee "$REPORT"

echo ""
echo ">>> Report saved to: $(pwd)/$REPORT"
echo ">>> Share this file's contents back to continue."
