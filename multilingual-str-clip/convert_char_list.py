"""
convert_char_list.py
Converts a plain-text character list (one character per line — the
common CRNN/CTC-style vocab format) into the JSON array format vocab.py's
Vocab.load() expects.

Usage (single file):
    python convert_char_list.py --input crnn_hindi_char_list.txt --output vocab/hindi_vocab.json

Usage (batch, all txt files in a directory):
    python convert_char_list.py --input_dir char_lists/ --output_dir vocab/
"""
import argparse
import json
import os


def convert(input_path, output_path):
    chars = []
    seen = set()
    skipped_blank = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n").rstrip("\r")
            if not ch:
                skipped_blank += 1
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    dupes = 0
    with open(input_path, "r", encoding="utf-8") as f:
        total_nonblank = sum(1 for line in f if line.rstrip("\n").rstrip("\r"))
    dupes = total_nonblank - len(chars)

    print(f"{os.path.basename(input_path):40s} -> {output_path}  "
          f"({len(chars)} unique chars"
          + (f", {dupes} duplicate line(s) dropped" if dupes else "")
          + (f", {skipped_blank} blank line(s) skipped" if skipped_blank else "")
          + ")")
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Single input .txt file")
    ap.add_argument("--output", help="Single output .json path (required with --input)")
    ap.add_argument("--input_dir", help="Batch mode: convert every .txt file in this directory")
    ap.add_argument("--output_dir", help="Batch mode: write <name>_vocab.json here for each input")
    args = ap.parse_args()

    if args.input:
        if not args.output:
            raise ValueError("--output is required with --input")
        convert(args.input, args.output)
    elif args.input_dir:
        if not args.output_dir:
            raise ValueError("--output_dir is required with --input_dir")
        txt_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))
        for fname in txt_files:
            base = os.path.splitext(fname)[0]
            out_name = f"{base}_vocab.json" if not base.endswith("_vocab") else f"{base}.json"
            convert(os.path.join(args.input_dir, fname), os.path.join(args.output_dir, out_name))
    else:
        raise ValueError("Pass either --input/--output or --input_dir/--output_dir")


if __name__ == "__main__":
    main()
