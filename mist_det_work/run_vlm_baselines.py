#!/usr/bin/env python3
"""
run_vlm_baselines.py

Runs closed-source commercial vision-language-model baselines (GPT-4V,
Claude) on the same MIST recognition_gt crops used for every other
baseline, adding prediction columns to the SAME manifest.csv that
run_mist_baselines.py and build_mist_recognition_gt.py already write to
-- so all of Google Vision (pseudo-GT), Tesseract, PaddleOCR, EasyOCR,
GPT-4V, and Claude end up as columns on one file, ready for a single
scoring pass.

Mirrors BSTD's own closed-source-commercial-baseline methodology
(Table 9): a structured system+user prompt asking the model to return
the extracted text as a list, rather than freeform description. Since
MIST's ground-truth polygons are per-word instances, each crop should
normally contain a single word/short phrase -- multiple returned items
(rare, e.g. if a crop caught more than one word) are joined with spaces
into a single text field, same convention as every other engine here.

Neither model needs a plain "OCR mode" -- both are asked the same way
BSTD asked GPT-4: "Extract all text from this image. Return all words
in a list format like this: ['word1', 'word2']", with a system message
"You are an OCR system that extracts text from images."

Auth (never pasted into chat -- set as local environment variables):
    OPENAI_API_KEY      for --engine gpt4v   (pip install openai)
    ANTHROPIC_API_KEY   for --engine claude  (pip install anthropic)

Usage:
    conda activate mist_det
    cd ~/mist_det_work
    python3 run_vlm_baselines.py --engine gpt4v  --model gpt-4o
    python3 run_vlm_baselines.py --engine claude --model claude-sonnet-4-5
    # Adjust --model if the default below is stale by the time you run
    # this -- whatever vision-capable model your account has access to.
"""
import argparse
import ast
import base64
import csv
import os
import re
import time

MANIFEST_PATH = os.path.expanduser("~/mist_det_work/recognition_gt/manifest.csv")

SYSTEM_PROMPT = "You are an OCR system that extracts text from images."
USER_PROMPT = (
    "Extract all text from this image. Return all words in a list "
    "format like this: ['word1', 'word2']"
)


def parse_list_response(text):
    """The model is asked to return a Python-list-looking string.
    Try to actually parse it as one; fall back to pulling quoted
    substrings (only when the response actually looks list-shaped);
    fall back to the raw text if neither works."""
    text = text.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return " ".join(str(x) for x in parsed).strip()
    except (ValueError, SyntaxError):
        pass
    # Only attempt the quoted-substring fallback when the response
    # actually starts like the requested ['word1', 'word2'] list
    # format. Without this guard, the regex misfires on ordinary
    # English contractions in prose -- e.g. a refusal sentence like
    # "I'm sorry, but I can't extract text from this image." gets
    # mis-sliced into "m sorry, but I can" (the text between the
    # apostrophe in "I'm" and the one in "can't"), silently corrupting
    # what should have been an obvious, easy-to-filter refusal into
    # something that looks like a short, plausible OCR guess. A prose
    # response never starts with '[', so this check cleanly separates
    # "the model returned a list-ish string ast.literal_eval choked on
    # for a trivial reason" from "the model returned prose".
    if text.startswith("["):
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        if quoted:
            words = [a or b for a, b in quoted]
            return " ".join(w for w in words if w).strip()
    return text


def image_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64


def run_gpt4v(rows, model, checkpoint_every=100):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    for i, row in enumerate(rows):
        try:
            b64 = image_to_data_url(row["crop_path"])
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content
            text = parse_list_response(raw)
            row["gpt4v_text"] = text
            row["gpt4v_status"] = "ok" if text else "no_text_detected"
        except Exception as e:
            row["gpt4v_text"], row["gpt4v_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


def run_claude(rows, model, checkpoint_every=100):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    for i, row in enumerate(rows):
        try:
            b64 = image_to_data_url(row["crop_path"])
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": b64}},
                    {"type": "text", "text": USER_PROMPT},
                ]}],
            )
            raw = "".join(
                block.text for block in resp.content
                if getattr(block, "type", None) == "text"
            )
            text = parse_list_response(raw)
            row["claude_text"] = text
            row["claude_status"] = "ok" if text else "no_text_detected"
        except Exception as e:
            row["claude_text"], row["claude_status"] = "", f"exception:{e}"
        if (i + 1) % checkpoint_every == 0:
            yield i + 1, len(rows)
    yield len(rows), len(rows)


RUNNERS = {"gpt4v": run_gpt4v, "claude": run_claude}
DEFAULT_MODELS = {"gpt4v": "gpt-4o", "claude": "claude-sonnet-4-5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(RUNNERS.keys()), required=True)
    ap.add_argument("--model", default=None,
                     help="Overrides the built-in default if it's stale "
                          "by the time you run this.")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.engine]

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

    todo = [r for r in rows if r[status_col] not in ("ok", "no_text_detected")]
    print(f"{args.engine} ({model}): {len(rows) - len(todo)} already done, "
          f"{len(todo)} remaining.")

    def checkpoint():
        # Atomic + encoding-safe: write to a sibling .tmp file first, then
        # os.replace() it over the real manifest. This guarantees the
        # existing manifest.csv is never left truncated/corrupted if the
        # write is interrupted for ANY reason -- a crash, Ctrl-C, disk
        # full, power loss. Previously a mid-write exception (e.g. a VLM
        # response containing a stray UTF-16 surrogate -- 'utf-8' refuses
        # to encode those) truncated manifest.csv, silently dropping every
        # row after the bad one, including already-completed results from
        # OTHER engines further down the file. errors="replace" also means
        # a bad character now degrades to a literal "?" in that one cell
        # instead of crashing the whole run.
        tmp_path = MANIFEST_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, MANIFEST_PATH)

    start = time.time()
    for done, total in RUNNERS[args.engine](todo, model, args.checkpoint_every):
        elapsed = time.time() - start
        print(f"  [{done}/{total}] ({elapsed:.0f}s elapsed)")
        checkpoint()

    checkpoint()
    statuses = {}
    for r in rows:
        statuses[r[status_col]] = statuses.get(r[status_col], 0) + 1
    print(f"\n{args.engine} done. Status breakdown: {statuses}")
    print(f"Manifest updated: {MANIFEST_PATH}")
    print("Note: this is a per-call PAID API -- costs scale with the "
          f"{len(todo)} calls just made. Check your account's usage "
          "dashboard if you want to track spend.")


if __name__ == "__main__":
    main()
