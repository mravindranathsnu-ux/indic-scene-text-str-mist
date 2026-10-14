#!/usr/bin/env python3
"""
run_mist_baselines.py

Runs open-source OCR baselines on the same MIST recognition_gt crops used
for the Google Vision pseudo-GT (recognition_gt/manifest.csv), mirroring
BSTD's Table 9/Fig. 8 baseline set: Tesseract, SuryaOCR, PaddleOCR,
EasyOCR. Predictions are added as new columns on the SAME manifest (one
column pair per engine: <engine>_text, <engine>_status), scored later
against the gcv_text column by score_mist_baselines.py.

Google Vision itself is deliberately NOT re-scored here -- it's the
pseudo-ground-truth source for this eval set, so including it as a
"baseline" would be circular (it would trivially score 100%). This
mirrors the user's explicit decision (Sep 18 2026) to keep it unscored
and compare every OTHER system against it, same underlying idea as
BSTD's Table 9 but without re-including the reference system as a row.

Each engine only runs on crops whose language it actually supports --
unsupported language/engine combinations are recorded as
"unsupported_language" and left out of scoring, the same convention BSTD
itself uses ("NA" in their Fig. 8) rather than silently skipping or
crashing. This is itself a reportable finding: several open-source
engines don't cover most Indic scripts at all.

No retraining/preprocessing tricks are applied here (crops are used
exactly as extracted for Google Vision) so every system, including the
eventual CLIP/VLM tracks, is compared on identical input -- apples to
apples.

Setup (install only what you intend to actually run):
    pip install pytesseract                      # + system: sudo apt install tesseract-ocr tesseract-ocr-ben tesseract-ocr-guj tesseract-ocr-hin tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-mar tesseract-ocr-ori tesseract-ocr-pan tesseract-ocr-tam tesseract-ocr-tel
    pip install paddlepaddle paddleocr
    pip install easyocr
    pip install surya-ocr

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    python3 run_mist_baselines.py --engine tesseract
    python3 run_mist_baselines.py --engine paddleocr
    python3 run_mist_baselines.py --engine easyocr
    python3 run_mist_baselines.py --engine suryaocr
    # each is independent/resumable -- run whichever engines you have
    # installed, in any order, any number of times.
"""
import argparse
import csv
import os
import time

MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")

# --- Per-engine language-code maps --------------------------------------
# Our own language tags (from MIST's own "script" field) -> each engine's
# language identifier. A language absent from a map is simply not
# supported by that engine -- recorded as "unsupported_language".

TESSERACT_LANG = {
    "bengali": "ben", "gujarati": "guj", "hindi": "hin", "kannada": "kan",
    "malayalam": "mal", "marathi": "mar", "odia": "ori", "punjabi": "pan",
    "tamil": "tam", "telugu": "tel",
}

# PaddleOCR's Indic coverage is narrow in practice (matches BSTD Fig. 8's
# frequent "NA" for this engine) -- only what its multilingual model zoo
# actually ships is listed; everything else records unsupported_language.
PADDLEOCR_LANG = {
    "hindi": "hi", "marathi": "hi",  # PaddleOCR's "hi" model covers the
                                       # Devanagari script broadly.
    "tamil": "ta", "telugu": "te", "kannada": "ka",
}

EASYOCR_LANG = {
    "bengali": "bn", "hindi": "hi",
    "tamil": "ta", "telugu": "te",
    # marathi ("mr") deliberately excluded: EasyOCR 1.7.2's downloaded
    # Marathi recognition checkpoint has a 143-vs-127-class mismatch
    # against what the installed library code expects -- reproducible
    # even with a single-language reader, so this is a packaging bug in
    # that release, not a fixable calling-code issue. Recorded as
    # unsupported_language, same convention as a language the engine
    # genuinely never shipped.
}

# Surya's language codes follow ISO 639-1 and cover more Indic scripts
# than the others -- attempted for all 10; construction failure for any
# still degrades to unsupported_language rather than crashing.
SURYA_LANG = {
    "bengali": "bn", "gujarati": "gu", "hindi": "hi", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "odia": "or", "punjabi": "pa",
    "tamil": "ta", "telugu": "te",
}


def run_tesseract(rows, checkpoint_every=100):
    import pytesseract
    for i, row in enumerate(rows):
        lang = TESSERACT_LANG.get(row["language"])
        if lang is None:
            row["tesseract_text"], row["tesseract_status"] = "", "unsupported_language"
        else:
            try:
                text = pytesseract.image_to_string(row["crop_path"], lang=lang).strip()
                row["tesseract_text"] = text
                row["tesseract_status"] = "ok" if text else "no_text_detected"
            except Exception as e:
                row["tesseract_text"], row["tesseract_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


def _paddleocr_extract_text(result):
    """PaddleOCR's output shape changed between the 2.x (.ocr(), nested
    list-of-lines) and 3.x (.predict(), dict-like OCRResult with
    rec_texts) APIs. Handle both rather than assuming a version."""
    texts = []
    for item in result:
        if isinstance(item, dict) and "rec_texts" in item:
            texts.extend(item["rec_texts"])
        elif hasattr(item, "get") and item.get("rec_texts"):
            texts.extend(item["rec_texts"])
        elif hasattr(item, "rec_texts"):
            texts.extend(item.rec_texts or [])
        elif isinstance(item, list):
            # 2.x shape: result[0] is a list of [box, (text, score)] lines.
            for line in item:
                try:
                    texts.append(line[1][0])
                except (IndexError, TypeError):
                    pass
    return " ".join(t for t in texts if t).strip()


def run_paddleocr(rows, checkpoint_every=100):
    from paddleocr import PaddleOCR
    readers = {}
    for i, row in enumerate(rows):
        lang = PADDLEOCR_LANG.get(row["language"])
        if lang is None:
            row["paddleocr_text"], row["paddleocr_status"] = "", "unsupported_language"
        else:
            try:
                if lang not in readers:
                    # No show_log/use_angle_cls -- both removed/renamed in
                    # PaddleOCR 3.x. enable_mkldnn=False works around a
                    # PaddlePaddle 3.x PIR/oneDNN backend bug on CPU
                    # ("ConvertPirAttribute2RuntimeAttribute not
                    # support") -- costs some speed, not correctness.
                    try:
                        readers[lang] = PaddleOCR(lang=lang, enable_mkldnn=False)
                    except TypeError:
                        readers[lang] = PaddleOCR(lang=lang)
                reader = readers[lang]
                try:
                    result = reader.predict(row["crop_path"])  # 3.x API
                except AttributeError:
                    result = reader.ocr(row["crop_path"])  # 2.x API fallback
                text = _paddleocr_extract_text(result)
                row["paddleocr_text"] = text
                row["paddleocr_status"] = "ok" if text else "no_text_detected"
            except Exception as e:
                row["paddleocr_text"], row["paddleocr_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


def run_easyocr(rows, checkpoint_every=100):
    import easyocr
    readers = {}
    for i, row in enumerate(rows):
        lang = EASYOCR_LANG.get(row["language"])
        if lang is None:
            row["easyocr_text"], row["easyocr_status"] = "", "unsupported_language"
        else:
            try:
                if lang not in readers:
                    # lang-only, not [lang, "en"] -- pairing some Indic
                    # languages (observed: Marathi) with English hits a
                    # checkpoint/vocab size mismatch in EasyOCR's model
                    # zoo. lang-alone avoids that whole class of failure.
                    readers[lang] = easyocr.Reader([lang], gpu=True)
                result = readers[lang].readtext(row["crop_path"], detail=0)
                text = " ".join(result).strip()
                row["easyocr_text"] = text
                row["easyocr_status"] = "ok" if text else "no_text_detected"
            except Exception as e:
                row["easyocr_text"], row["easyocr_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


def run_suryaocr(rows, checkpoint_every=100):
    # Surya's RecognitionPredictor.__call__ signature has changed across
    # releases (some versions require a DetectionPredictor positionally,
    # some accept it only as a keyword, some don't need one at all when
    # you already have tight crops and don't need it to also do
    # detection). Try progressively simpler call shapes rather than
    # assuming one -- the installed 0.22.1 rejected the 4-positional-arg
    # form ("takes from 2 to 3 positional arguments but 4 were given").
    from surya.recognition import RecognitionPredictor
    from PIL import Image

    rec_predictor = RecognitionPredictor()
    det_predictor = None
    try:
        from surya.detection import DetectionPredictor
        det_predictor = DetectionPredictor()
    except Exception:
        pass  # fine -- only needed if the call shape below requires it

    def call_recognizer(img, lang):
        attempts = [
            lambda: rec_predictor([img], [[lang]]),               # images, langs
            lambda: rec_predictor([img], langs=[[lang]]),          # langs as kwarg
            lambda: rec_predictor([img], [[lang]], det_predictor), # older 3-positional-arg form
            lambda: rec_predictor([img]),                          # no lang hint, auto-detect
        ]
        last_err = None
        for attempt in attempts:
            try:
                return attempt()
            except Exception as e:
                last_err = e
        raise last_err

    for i, row in enumerate(rows):
        lang = SURYA_LANG.get(row["language"])
        if lang is None:
            row["suryaocr_text"], row["suryaocr_status"] = "", "unsupported_language"
        else:
            try:
                img = Image.open(row["crop_path"]).convert("RGB")
                predictions = call_recognizer(img, lang)
                text = ""
                if predictions and getattr(predictions[0], "text_lines", None):
                    text = " ".join(l.text for l in predictions[0].text_lines).strip()
                row["suryaocr_text"] = text
                row["suryaocr_status"] = "ok" if text else "no_text_detected"
            except Exception as e:
                row["suryaocr_text"], row["suryaocr_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


RUNNERS = {
    "tesseract": run_tesseract,
    "paddleocr": run_paddleocr,
    "easyocr": run_easyocr,
    "suryaocr": run_suryaocr,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(RUNNERS.keys()), required=True)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    args = ap.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    text_col = f"{args.engine}_text"
    status_col = f"{args.engine}_status"
    if text_col not in fieldnames:
        fieldnames += [text_col, status_col]
    for row in rows:
        row.setdefault(text_col, "")
        row.setdefault(status_col, "pending")

    # Resumable: skip rows already labeled by this engine.
    todo = [r for r in rows if r[status_col] not in ("ok", "no_text_detected", "unsupported_language")]
    print(f"{args.engine}: {len(rows) - len(todo)} already done, {len(todo)} remaining.")

    def checkpoint():
        # Atomic + encoding-safe write -- see run_vlm_baselines.py for why:
        # a mid-write crash (e.g. a stray character 'utf-8' can't encode)
        # used to truncate manifest.csv in place, dropping rows silently.
        tmp_path = MANIFEST_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, MANIFEST_PATH)

    start = time.time()
    for done, total in RUNNERS[args.engine](todo, args.checkpoint_every):
        elapsed = time.time() - start
        print(f"  [{done}/{total}] ({elapsed:.0f}s elapsed)")
        checkpoint()

    checkpoint()
    statuses = {}
    for r in rows:
        statuses[r[status_col]] = statuses.get(r[status_col], 0) + 1
    print(f"\n{args.engine} done. Status breakdown: {statuses}")
    print(f"Manifest updated: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
