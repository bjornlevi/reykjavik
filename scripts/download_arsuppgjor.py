#!/usr/bin/env python3
"""Download arsuppgjor CSVs from the Reykjavik open data portal."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, unquote

import requests

DATASET_URL = "https://gagnagatt.reykjavik.is/dataset/arsuppgjor"
DOWNLOAD_RE = re.compile(r'href="([^"]+download/[^"]+\.csv)"', re.IGNORECASE)


def _extract_download_urls(html: str, base_url: str) -> list[str]:
    urls = [urljoin(base_url, match) for match in DOWNLOAD_RE.findall(html)]
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return unquote(name) or "download.csv"


def _download(urls: Iterable[str], raw_dir: Path, force: bool, timeout: int) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "reykjavik-arsuppgjor-pipeline/1.0"})

    results: list[dict] = []
    for url in urls:
        filename = _filename_from_url(url)
        dest = raw_dir / filename

        if dest.exists() and not force:
            results.append({
                "url": url,
                "filename": filename,
                "status": "skipped",
                "bytes": dest.stat().st_size,
            })
            continue

        tmp_path = dest.with_suffix(dest.suffix + ".part")
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                bytes_written = 0
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)

                tmp_path.replace(dest)

                results.append({
                    "url": url,
                    "filename": filename,
                    "status": "downloaded",
                    "bytes": bytes_written,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                })
        except requests.RequestException as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            results.append({
                "url": url,
                "filename": filename,
                "status": f"error: {exc}",
                "bytes": 0,
            })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Reykjavik arsuppgjor CSVs")
    parser.add_argument("--dataset-url", default=DATASET_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(args.dataset_url, timeout=args.timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch dataset page: {exc}", file=sys.stderr)
        return 1

    urls = _extract_download_urls(response.text, args.dataset_url)
    if not urls:
        print("No CSV download URLs found on dataset page.", file=sys.stderr)
        return 1

    if args.dry_run:
        for url in urls:
            print(url)
        return 0

    results = _download(urls, raw_dir, args.force, args.timeout)

    manifest = {
        "dataset_url": args.dataset_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "files": results,
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))

    errors = [item for item in results if item["status"].startswith("error")]
    if errors:
        print(f"Completed with {len(errors)} errors. See manifest.json for details.")
        return 2

    print(f"Downloaded {len(results)} files into {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
