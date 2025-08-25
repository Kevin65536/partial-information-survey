#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch all citations of "Nonnegative Decomposition of Multivariate Information" (Williams & Beer, 2010)
from Semantic Scholar Graph API, and export titles & abstracts to JSON only.

Preconfigured stable defaults:
- Backend: Semantic Scholar only
- API Key: embedded (can be overridden by env S2_API_KEY or SEMANTIC_SCHOLAR_API_KEY)
- Pagination: limit=100, sleep=0.6s
- Detail enrichment: batch via /paper/batch with size=50
- Proxies: disabled by default (mimic browser direct TLS)

Usage (PowerShell):
    python "d:\workspace\document\pid method survey\script\fetch_wb_citations.py" ^
        --paper-title "Nonnegative Decomposition of Multivariate Information" ^
        --out-dir "d:\workspace\document\pid method survey"
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Stable defaults for this workspace
EMBEDDED_S2_API_KEY = "GypJBCAUeB6Lj5iE4ae5f2vi9NzSOHHbaIhmMHH0"
DEFAULT_S2_LIMIT = 100
DEFAULT_S2_SLEEP = 0.6
DEFAULT_S2_BATCH_SIZE = 50
DISABLE_PROXIES = True

# Global session configured in main()
SESSION: Optional[requests.Session] = None


def get_headers() -> Dict[str, str]:
    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or EMBEDDED_S2_API_KEY
    ua_email = os.environ.get("USER_EMAIL") or os.environ.get("OPENALEX_MAILTO") or ""
    headers = {
        "Accept": "application/json",
        # A friendly UA helps some endpoints and corporate gateways
        "User-Agent": f"partial-information-survey/0.1 ({ua_email})".strip(),
    }
    if api_key:
        # Header name per official docs
        headers["x-api-key"] = api_key
    return headers


def http_get(url: str, params: Optional[Dict] = None, max_retries: int = 6, backoff: float = 1.2) -> dict:
    headers = get_headers()
    sess = SESSION or requests
    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff * attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            # Respect rate limit/backoff
            if os.environ.get("PID_FETCH_DEBUG") == "1":
                try:
                    print(f"[DEBUG] S2 {resp.status_code} at {url} | headers={dict(resp.headers)} | body={resp.text[:200]}…")
                except Exception:
                    pass
            time.sleep(backoff * attempt)
            continue
        # Other client/server errors -> raise
        try:
            payload = resp.json()
        except Exception:
            payload = {"text": resp.text}
        raise RuntimeError(f"GET {url} failed: {resp.status_code} {payload}")
    raise RuntimeError(f"GET {url} exceeded retries")


def search_paper_by_title(title: str) -> Optional[dict]:
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": title,
        "limit": 5,
        "fields": "paperId,title,year,authors,venue,externalIds,url",
    }
    data = http_get(url, params)
    candidates = data.get("data", [])
    if not candidates:
        return None
    # Prefer exact (case-insensitive) title match if available
    title_l = title.strip().lower()
    exact = [p for p in candidates if (p.get("title") or "").strip().lower() == title_l]
    if exact:
        return exact[0]
    # Otherwise return top result
    return candidates[0]


def fetch_all_citations(paper_id: str, limit: int = 100, sleep: float = 0.3, basic_fields: bool = False) -> List[dict]:
    url = f"{BASE_URL}/paper/{paper_id}/citations"
    offset = 0
    all_rows: List[dict] = []
    # Always fetch minimal fields for stability; details will be enriched via batch endpoint
    fields = ",".join([
        "citingPaper.paperId",
        "citingPaper.title",
        "citingPaper.year",
        "citingPaper.url",
    ])
    while True:
        params = {"offset": offset, "limit": limit, "fields": fields}
        data = http_get(url, params)
        rows = data.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        offset += len(rows)
        # polite pacing
        time.sleep(sleep)
        if len(rows) < limit:
            break
    return all_rows


def s2_batch_fetch_details(paper_ids: List[str], fields: List[str], batch_size: int = 100, sleep: float = 0.2,
                           max_retries: int = 5, backoff: float = 1.2) -> Dict[str, dict]:
    """Fetch detailed fields for a list of S2 paper IDs using /paper/batch endpoint.
    Returns a dict mapping paperId -> paper dict with requested fields.
    """
    url = f"{BASE_URL}/paper/batch"
    params = {"fields": ",".join(fields)}
    headers = get_headers()
    headers["Content-Type"] = "application/json"
    sess = SESSION or requests
    out: Dict[str, dict] = {}

    for i in range(0, len(paper_ids), batch_size):
        chunk = paper_ids[i:i + batch_size]
        payload = {"ids": chunk}
        for attempt in range(1, max_retries + 1):
            try:
                resp = sess.post(url, params=params, headers=headers, json=payload, timeout=60)
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(backoff * attempt)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for p in data:
                        pid = p.get("paperId")
                        if pid:
                            out[pid] = p
                break
            if resp.status_code in (429, 500, 502, 503, 504):
                if os.environ.get("PID_FETCH_DEBUG") == "1":
                    try:
                        print(f"[DEBUG] S2 batch {resp.status_code} | body={resp.text[:200]}…")
                    except Exception:
                        pass
                time.sleep(backoff * attempt)
                continue
            # other errors
            try:
                payload_err = resp.json()
            except Exception:
                payload_err = {"text": resp.text}
            raise RuntimeError(f"POST {url} failed: {resp.status_code} {payload_err}")
        time.sleep(sleep)
    return out


def flatten_citation_row(row: dict) -> dict:
    p = row.get("citingPaper") or {}
    authors = p.get("authors") or []
    ext = p.get("externalIds") or {}
    return {
        "s2PaperId": p.get("paperId") or "",
        "title": (p.get("title") or "").replace("\n", " ").strip(),
        "abstract": (p.get("abstract") or "").replace("\n", " ").strip(),
        "year": p.get("year") or "",
        "venue": p.get("venue") or "",
        "url": p.get("url") or "",
        "doi": ext.get("DOI") or ext.get("doi") or "",
        "arxivId": ext.get("ArXiv") or ext.get("arXiv") or "",
    "authors": ", ".join([a.get("name", "") for a in authors if a.get("name")]),
    "citationCount": p.get("citationCount") or 0,
    }


def write_json(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


## CSV/Markdown outputs removed per requirement


## OpenAlex fallback removed per requirement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-title",
        type=str,
        default="Nonnegative Decomposition of Multivariate Information",
        help="Title to search on Semantic Scholar",
    )
    parser.add_argument(
        "--paper-id", type=str, default=None, help="Optional Semantic Scholar paperId to skip search"
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--basename", type=str, default="citations_wb2010", help="Base filename without extension"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Configure session: ignore env proxies by default for stable direct TLS
    global SESSION
    SESSION = requests.Session()
    if DISABLE_PROXIES:
        SESSION.trust_env = False
        SESSION.proxies = {}

    if args.paper_id:
        paper = {"paperId": args.paper_id, "title": args.paper_title}
    else:
        paper = search_paper_by_title(args.paper_title)
        if not paper:
            print(f"[ERROR] Could not find paper by title: {args.paper_title}", file=sys.stderr)
            sys.exit(1)

    flat: List[dict] = []
    try:
        pid = paper["paperId"]
        print(
            f"[INFO] [S2] Target paperId: {pid} | title: {paper.get('title')} | year: {paper.get('year')}"
        )
        raw = fetch_all_citations(pid, limit=DEFAULT_S2_LIMIT, sleep=DEFAULT_S2_SLEEP, basic_fields=True)
        print(f"[INFO] [S2] Fetched citation records (raw): {len(raw)}")
        # Extract citingPaper.paperId list
        ids = []
        for r in raw:
            cp = r.get("citingPaper") or {}
            pid2 = cp.get("paperId")
            if pid2:
                ids.append(pid2)
        ids = list(dict.fromkeys(ids))  # dedup preserve order
        print(f"[INFO] [S2] Enriching details via batch for {len(ids)} papers…")
        fields = [
            "paperId","title","abstract","year","venue","url","externalIds","authors","citationCount"
        ]
        details = s2_batch_fetch_details(ids, fields=fields, batch_size=DEFAULT_S2_BATCH_SIZE, sleep=0.3)
        # Build synthetic rows compatible with flatten_citation_row output structure
        nested = []
        for pid2 in ids:
            p = details.get(pid2)
            if not p:
                continue
            nested.append({"citingPaper": p})
        # Flatten to standard row shape
        flat = [flatten_citation_row(r) for r in nested]
    except Exception as e:
        print(f"[ERROR] Semantic Scholar fetch failed: {e}", file=sys.stderr)
        sys.exit(1)
    # Drop entries without title (rare)
    flat = [r for r in flat if r.get("title")]
    # Optional: de-duplicate by s2PaperId/title
    seen = set()
    dedup = []
    for r in flat:
        key = r.get("s2PaperId") or r.get("title").lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    print(f"[INFO] After de-dup: {len(dedup)}")

    json_path = os.path.join(args.out_dir, f"{args.basename}.json")

    write_json(dedup, json_path)

    print(f"[OK] JSON: {json_path}")


if __name__ == "__main__":
    main()
