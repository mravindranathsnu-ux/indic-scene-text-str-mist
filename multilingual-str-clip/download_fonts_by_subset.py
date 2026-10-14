#!/usr/bin/env python3
"""
download_fonts_by_subset.py
Downloads EVERY Google Fonts family that supports a given script (subset),
fully automated -- no manual browsing or zip downloads. Requires a free
Google Fonts API key (no billing needed): get one at
https://developers.google.com/fonts/docs/developer_api -> "Get a Key".

Usage:
    python download_fonts_by_subset.py --subset telugu --api_key YOUR_KEY \
        --out_dir incoming/fonts_telugu_raw
    # then feed that folder straight into the existing pipeline:
    python organize_fonts.py --language telugu --extracted_dir incoming/fonts_telugu_raw

Common Indic subset names to try: telugu, tamil, devanagari, gujarati,
kannada, malayalam, bengali, oriya, gurmukhi.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"


def fetch_family_list(api_key, sort="alpha"):
    url = f"{API_URL}?key={api_key}&sort={sort}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Google Fonts API request failed (HTTP {e.code}). "
            f"Check that --api_key is correct and the Fonts API is enabled for it.\n"
            f"Response: {body[:500]}"
        )
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach the Google Fonts API: {e}")
    return data.get("items", [])


def safe_filename(family, variant):
    fam = family.replace(" ", "")
    var = variant.replace(" ", "")
    return f"{fam}-{var}.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                     help="Script subset to filter by, e.g. telugu, tamil, devanagari, gurmukhi")
    ap.add_argument("--api_key", required=True, help="Free Google Fonts API key")
    ap.add_argument("--out_dir", required=True, help="Where to save downloaded .ttf files")
    ap.add_argument("--dry_run", action="store_true", help="List matching families/files, download nothing")
    ap.add_argument("--variants", nargs="+", default=None,
                     help="Only these variants (e.g. regular 700 italic). Default: all available per family.")
    args = ap.parse_args()

    print(f"Querying Google Fonts API for subset={args.subset!r} ...")
    families = fetch_family_list(args.api_key)
    print(f"Total families in catalog: {len(families)}")

    matched = [f for f in families if args.subset in f.get("subsets", [])]
    print(f"Families supporting {args.subset!r}: {len(matched)}")
    if not matched:
        available_subsets = sorted({s for f in families for s in f.get("subsets", [])})
        raise SystemExit(
            f"No families found for subset {args.subset!r}. Known subsets in this catalog include:\n"
            f"  {', '.join(available_subsets)}"
        )

    total_files = 0
    for fam in matched:
        variants = fam.get("files", {})
        if args.variants:
            variants = {k: v for k, v in variants.items() if k in args.variants}
        total_files += len(variants)

    print(f"Total font files to download: {total_files}")
    for fam in matched:
        variants = fam.get("files", {})
        if args.variants:
            variants = {k: v for k, v in variants.items() if k in args.variants}
        print(f"  {fam['family']}: {len(variants)} variant(s)")

    if args.dry_run:
        print("\n(dry run -- nothing downloaded. Re-run without --dry_run to actually download.)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    downloaded, failed = 0, []
    for fam in matched:
        variants = fam.get("files", {})
        if args.variants:
            variants = {k: v for k, v in variants.items() if k in args.variants}
        for variant, url in variants.items():
            url = url.replace("http://", "https://")  # gstatic supports https; avoid mixed-content issues
            fname = safe_filename(fam["family"], variant)
            dest = os.path.join(args.out_dir, fname)
            try:
                urllib.request.urlretrieve(url, dest)
                downloaded += 1
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                failed.append((fname, str(e)))
            time.sleep(0.05)  # light throttling, courteous to the CDN

    print(f"\nDownloaded {downloaded}/{total_files} font file(s) to {args.out_dir}/")
    if failed:
        print(f"!! {len(failed)} failed:")
        for fname, err in failed[:10]:
            print(f"   {fname}: {err}")

    print(f"\nNext step:")
    print(f"  python organize_fonts.py --language <language> --extracted_dir {args.out_dir}")


if __name__ == "__main__":
    main()
