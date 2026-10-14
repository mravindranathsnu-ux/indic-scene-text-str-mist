"""
utils.py
Evaluation metrics for scene text recognition:
  - CER / WER        : Character / Word Error Rate (edit-distance based)
  - CRR               : Character Recognition Rate = 1 - CER
  - WRR                : Word Recognition Rate = exact-string-match accuracy
                          (the standard "word accuracy" reported in STR papers —
                          each cropped image counts as fully correct or fully
                          wrong, no partial credit)
"""
import editdistance
import torch


def log_device(device, context=""):
    """Prints an unambiguous confirmation of whether GPU is actually being
    used — call this right after `device = torch.device(args.device)` in
    any entry-point script. Prefix with `context` (e.g. "train.py") when
    calling from multiple scripts to tell logs apart."""
    prefix = f"[{context}] " if context else ""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"{prefix}Using device: cuda:{idx}  ({name}, {total_gb:.1f} GB)")
    elif device.type == "mps":
        print(f"{prefix}Using device: mps (Apple Silicon GPU)")
    else:
        print(f"{prefix}Using device: cpu  "
              f"(no GPU detected/selected — training and inference will be much slower)")


def compute_cer(preds, refs):
    """Character Error Rate: total character edit distance / total reference
    characters, aggregated over the whole set (not averaged per-sample)."""
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        total_dist += editdistance.eval(p, r)
        total_len += max(len(r), 1)
    return total_dist / max(total_len, 1)


def compute_wer(preds, refs):
    """Word Error Rate: word-level edit distance / total reference words.
    Useful when a sample's label may be a multi-word line; for single-word
    crops this reduces to the same thing as (1 - WRR)."""
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        p_words, r_words = p.split(), r.split()
        total_dist += editdistance.eval(p_words, r_words)
        total_len += max(len(r_words), 1)
    return total_dist / max(total_len, 1)


def compute_character_recognition_rate(preds, refs):
    """Character Recognition Rate (CRR) = 1 - CER, expressed as an accuracy
    (higher is better) rather than an error rate."""
    return 1.0 - compute_cer(preds, refs)


def compute_word_recognition_rate(preds, refs):
    """Word Recognition Rate (WRR): fraction of samples whose predicted
    string exactly matches the ground truth, i.e. sequence-level accuracy.
    This is the primary headline metric used in most scene-text-recognition
    papers/benchmarks (distinct from 1 - WER, which gives partial credit
    for multi-word lines that partially match)."""
    if not refs:
        return 0.0
    correct = sum(1 for p, r in zip(preds, refs) if p == r)
    return correct / len(refs)


def compute_all_metrics(preds, refs):
    """Convenience wrapper returning every metric in one dict."""
    return {
        "CER": compute_cer(preds, refs),
        "WER": compute_wer(preds, refs),
        "CRR": compute_character_recognition_rate(preds, refs),
        "WRR": compute_word_recognition_rate(preds, refs),
        "num_samples": len(refs),
    }
