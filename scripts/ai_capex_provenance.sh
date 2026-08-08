#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import gzip
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PDF_URL = "https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf"
UA = "AI-capex-provenance-builder/1.0 (+public research archive retrieval)"


def fetch(url: str, *, headers: dict[str, str] | None = None, timeout: int = 90) -> bytes:
    h = {"User-Agent": UA, "Accept": "*/*"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        print(f"FETCH {r.status} {r.headers.get('content-type')} {len(data)} bytes {url}")
        return data


def show_json(label: str, url: str) -> None:
    print(f"\n===== {label} =====")
    try:
        raw = fetch(url, timeout=60)
        obj = json.loads(raw)
        print(json.dumps(obj, indent=2)[:30000])
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


show_json("OpenAlex", "https://api.openalex.org/works/doi:10.71468/P1RP4F")
show_json("DataCite", "https://api.datacite.org/dois/10.71468/P1RP4F")
show_json("Crossref", "https://api.crossref.org/works/10.71468/P1RP4F")
show_json("Unpaywall", "https://api.unpaywall.org/v2/10.71468/P1RP4F?email=research%40example.com")
show_json("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/DOI:10.71468/P1RP4F?fields=title,url,externalIds,openAccessPdf")
show_json("Wayback CDX", "https://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(PDF_URL, safe="") + "&output=json&filter=statuscode:200&fl=timestamp,original,mimetype,statuscode,digest,length&collapse=digest")

print("\n===== Common Crawl lookup =====")
found = False
try:
    collections = json.loads(fetch("https://index.commoncrawl.org/collinfo.json", timeout=60))
    print("Collections:", [c.get("id") for c in collections[:12]])
    for coll in collections[:16]:
        api = coll.get("cdx-api") or coll.get("index")
        if not api:
            continue
        query = api + "?url=" + urllib.parse.quote(PDF_URL, safe="") + "&output=json"
        try:
            raw = fetch(query, timeout=60).decode("utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"Collection {coll.get('id')} lookup error: {exc}")
            continue
        if not raw:
            continue
        print(f"Collection {coll.get('id')} records:\n{raw[:10000]}")
        for line in raw.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("status")) != "200":
                continue
            filename = rec.get("filename")
            offset = int(rec.get("offset", 0))
            length = int(rec.get("length", 0))
            if not filename or length <= 0:
                continue
            warc_url = "https://data.commoncrawl.org/" + filename
            print("Attempting WARC record:", warc_url, offset, length)
            try:
                compressed = fetch(
                    warc_url,
                    headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                    timeout=180,
                )
                record = gzip.GzipFile(fileobj=io.BytesIO(compressed)).read()
                # Locate the HTTP response and split its headers from its entity body.
                http_pos = record.find(b"HTTP/")
                if http_pos < 0:
                    raise RuntimeError("HTTP response header not found in WARC member")
                body_pos = record.find(b"\r\n\r\n", http_pos)
                sep = 4
                if body_pos < 0:
                    body_pos = record.find(b"\n\n", http_pos)
                    sep = 2
                if body_pos < 0:
                    raise RuntimeError("HTTP response body separator not found")
                payload = record[body_pos + sep :]
                pdf_pos = payload.find(b"%PDF-")
                if pdf_pos > 0:
                    payload = payload[pdf_pos:]
                if not payload.startswith(b"%PDF-"):
                    raise RuntimeError(f"Recovered payload is not a PDF; prefix={payload[:80]!r}")
                out = Path("LBNL_2026_from_Common_Crawl.pdf")
                out.write_bytes(payload)
                print(f"SUCCESS: wrote {out} ({out.stat().st_size} bytes) from {coll.get('id')}")
                found = True
                break
            except Exception as exc:
                print(f"WARC extraction error: {type(exc).__name__}: {exc}")
        if found:
            break
except Exception as exc:
    print(f"Common Crawl top-level error: {type(exc).__name__}: {exc}")

if not found:
    print("No recoverable Common Crawl PDF found in the queried collections.")
PY
