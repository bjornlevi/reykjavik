#!/usr/bin/env python3
"""Resolve vm_number values to entity names using Skatturinn fyrirtækjaskrá."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://www.skatturinn.is/fyrirtaekjaskra/leit"
KENNITALA_PATH = "/fyrirtaekjaskra/leit/kennitala/"


def _load_vm_numbers(parquet_path: Path, column: str | None = None) -> list[str]:
    if column is None:
        # Prefer vm_number, fall back to vm_numer (seen in source data)
        schema = pd.read_parquet(parquet_path, engine="pyarrow").columns
        if "vm_number" in schema:
            column = "vm_number"
        elif "vm_numer" in schema:
            column = "vm_numer"
        else:
            raise KeyError("Neither vm_number nor vm_numer found in parquet.")
    df = pd.read_parquet(parquet_path, columns=[column])
    series = df[column].dropna().astype(str).str.strip()
    series = series[series != ""]
    # Keep digits only for lookup, but preserve unique values
    series = series.str.replace(r"\D+", "", regex=True)
    return sorted(series.unique().tolist())


def _fetch(session: requests.Session, url: str, timeout: int) -> requests.Response:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def _parse_no_result(soup: BeautifulSoup) -> bool:
    text = soup.get_text(" ", strip=True)
    return "skilaði engri niðurstöðu" in text.lower()


def _extract_kennitala_url(soup: BeautifulSoup) -> str | None:
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if KENNITALA_PATH in href:
            return urljoin(SEARCH_URL, href)
    return None


def _extract_name(soup: BeautifulSoup) -> str | None:
    # Skatturinn pages tend to use h1 for the entity name
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)
    return None


def lookup_vm_number(session: requests.Session, vm_number: str, timeout: int) -> dict:
    search_url = f"{SEARCH_URL}?nafn=&heimili=&kt=&vsknr={vm_number}"
    try:
        response = _fetch(session, search_url, timeout)
    except requests.RequestException as exc:
        return {
            "vm_number": vm_number,
            "status": f"error: {exc}",
            "name": None,
            "url": search_url,
        }

    soup = BeautifulSoup(response.text, "html.parser")
    if _parse_no_result(soup):
        return {
            "vm_number": vm_number,
            "status": "no_result",
            "name": None,
            "url": search_url,
        }

    # Try to find a direct kennitala link in the search results
    kennitala_url = _extract_kennitala_url(soup)
    if not kennitala_url:
        name = _extract_name(soup)
        return {
            "vm_number": vm_number,
            "status": "result_unknown",
            "name": name,
            "url": search_url,
        }

    try:
        details_response = _fetch(session, kennitala_url, timeout)
    except requests.RequestException as exc:
        return {
            "vm_number": vm_number,
            "status": f"error: {exc}",
            "name": None,
            "url": kennitala_url,
        }

    details_soup = BeautifulSoup(details_response.text, "html.parser")
    name = _extract_name(details_soup)
    return {
        "vm_number": vm_number,
        "status": "ok" if name else "missing_name",
        "name": name,
        "url": kennitala_url,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lookup vm_number entities via Skatturinn")
    parser.add_argument("--parquet", default="data/processed/arsuppgjor_combined.parquet")
    parser.add_argument("--column", default="", help="Override vm number column name")
    parser.add_argument("--out", default="data/processed/vm_entities.csv")
    parser.add_argument("--cache", default="data/processed/vm_entities.json")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25, help="Write cache every N lookups")
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Missing parquet file: {parquet_path}")
        return 1

    column = args.column.strip() or None
    try:
        vm_numbers = _load_vm_numbers(parquet_path, column=column)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.limit and args.limit > 0:
        vm_numbers = vm_numbers[: args.limit]

    cache_path = Path(args.cache)
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cached = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "reykjavik-arsuppgjor-lookup/1.0"})

    results = []
    total = len(vm_numbers)
    processed = 0
    saved_since = 0
    for vm_number in vm_numbers:
        if vm_number in cached:
            results.append(cached[vm_number])
            processed += 1
            continue
        result = lookup_vm_number(session, vm_number, args.timeout)
        cached[vm_number] = result
        results.append(result)
        processed += 1
        saved_since += 1
        if args.save_every > 0 and saved_since >= args.save_every:
            cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=True))
            saved_since = 0
        print(f"{processed}/{total} looked up", end="\r", flush=True)
        time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=True))

    print(f"\nWrote {out_path} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
