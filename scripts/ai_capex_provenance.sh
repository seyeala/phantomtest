#!/usr/bin/env bash
set -euo pipefail

ROOT="AI_Capex_Macro_Provenance_Package_2026-08-08"
RES="$ROOT/resources"
PROV="$ROOT/provenance"
OUTZIP="AI_Capex_Macro_Provenance_Package_2026-08-08.zip"

rm -rf "$ROOT" "$OUTZIP"
mkdir -p "$RES" "$PROV"

get() {
  local url="$1"
  local out="$2"
  echo "::group::Downloading $url"
  curl --fail --location --show-error --silent \
    --retry 6 --retry-all-errors --retry-delay 4 \
    --connect-timeout 30 --max-time 1200 \
    --user-agent 'Mozilla/5.0 AI-Capex-Provenance-Builder/1.0' \
    --output "$out" "$url"
  test -s "$out"
  stat --printf='%n: %s bytes\n' "$out"
  file "$out"
  echo "::endgroup::"
}

# Original issuer-hosted reports.
get \
  'https://iea.blob.core.windows.net/assets/3179f7f8-01f6-4dd6-bffa-c9f7b73f1dc9/KeyQuestionsonEnergyandAI.pdf' \
  "$RES/IEA_2026_Key_Questions_on_Energy_and_AI.pdf"

get \
  'https://iea.blob.core.windows.net/assets/de9dea13-b07d-42c5-a398-d1b3ae17d866/EnergyandAI.pdf' \
  "$RES/IEA_2025_Energy_and_AI.pdf"

# The official eScholarship endpoint applies an automated-browser challenge. The
# Internet Archive captured the exact official PDF on 2026-07-26. We preserve
# the archive record and retrieve the binary with the `id_` modifier, which
# returns the original response body without Wayback rewriting.
LBNL_OFFICIAL='https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf'
LBNL_CDX='https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Fescholarship.org%2Fcontent%2Fqt33m6w3x0%2Fqt33m6w3x0.pdf&output=json&filter=statuscode%3A200&fl=timestamp%2Coriginal%2Cmimetype%2Cstatuscode%2Cdigest%2Clength&collapse=digest'
LBNL_ARCHIVE='https://web.archive.org/web/20260726103937id_/https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf'
get "$LBNL_CDX" "$PROV/LBNL_2026_Wayback_CDX_Record.json"
get "$LBNL_ARCHIVE" "$RES/LBNL_2026_United_States_Data_Center_Energy_Usage_Report_2025_Update.pdf"

get \
  'https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report_1.pdf' \
  "$RES/LBNL_2024_United_States_Data_Center_Energy_Usage_Report.pdf"

get \
  'https://www.federalreserve.gov/monetarypolicy/files/20260710_mprfullreport.pdf' \
  "$RES/Federal_Reserve_2026_Monetary_Policy_Report_July.pdf"

FED_NOTE_URL='https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html'
get "$FED_NOTE_URL" "$RES/Federal_Reserve_2026_The_Global_Trade_Effects_of_the_AI_Infrastructure_Boom.html"

# Produce a portable visual snapshot of the official web-only Federal Reserve
# note. The HTML source remains in the package as the source-preserving copy.
CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)"
if [[ -z "$CHROME" ]]; then
  echo 'No Chrome/Chromium executable found.' >&2
  exit 1
fi
"$CHROME" --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --virtual-time-budget=20000 \
  --print-to-pdf="$RES/Federal_Reserve_2026_The_Global_Trade_Effects_of_the_AI_Infrastructure_Boom.pdf" \
  "$FED_NOTE_URL"
test -s "$RES/Federal_Reserve_2026_The_Global_Trade_Effects_of_the_AI_Infrastructure_Boom.pdf"

python3 -m pip install --quiet pypdf

python3 <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

root = Path('AI_Capex_Macro_Provenance_Package_2026-08-08')
resources = root / 'resources'
retrieved = '2026-08-08'

sources = [
    {
        'filename': 'IEA_2026_Key_Questions_on_Energy_and_AI.pdf',
        'title': 'Key Questions on Energy and AI',
        'institution': 'International Energy Agency',
        'publication_date': '2026-04-16',
        'document_type': 'Official report PDF',
        'official_landing_url': 'https://www.iea.org/reports/key-questions-on-energy-and-ai',
        'official_source_url': 'https://iea.blob.core.windows.net/assets/3179f7f8-01f6-4dd6-bffa-c9f7b73f1dc9/KeyQuestionsonEnergyandAI.pdf',
        'retrieval_url': 'https://iea.blob.core.windows.net/assets/3179f7f8-01f6-4dd6-bffa-c9f7b73f1dc9/KeyQuestionsonEnergyandAI.pdf',
        'doi': '',
        'license_note': 'See the report copyright page and IEA terms; the report states a Creative Commons Attribution 4.0 licence.',
        'role_in_model': 'Primary global update for data-center investment, electricity demand, AI-focused capacity, and physical bottlenecks.',
        'representation': 'Complete original issuer-hosted PDF',
        'expected_pages_min': 130,
        'expected_pages_max': 150,
        'title_checks': ['Key Questions on Energy and AI', 'International Energy Agency'],
    },
    {
        'filename': 'IEA_2025_Energy_and_AI.pdf',
        'title': 'Energy and AI',
        'institution': 'International Energy Agency',
        'publication_date': '2025-04-10',
        'document_type': 'Official report PDF',
        'official_landing_url': 'https://www.iea.org/reports/energy-and-ai',
        'official_source_url': 'https://iea.blob.core.windows.net/assets/de9dea13-b07d-42c5-a398-d1b3ae17d866/EnergyandAI.pdf',
        'retrieval_url': 'https://iea.blob.core.windows.net/assets/de9dea13-b07d-42c5-a398-d1b3ae17d866/EnergyandAI.pdf',
        'doi': '',
        'license_note': 'See the report copyright page and IEA terms; the report states a Creative Commons Attribution 4.0 licence.',
        'role_in_model': 'Foundational global physical model for data-center demand, energy infrastructure, and AI scenarios.',
        'representation': 'Complete original issuer-hosted PDF',
        'expected_pages_min': 290,
        'expected_pages_max': 320,
        'title_checks': ['Energy and AI', 'International Energy Agency'],
    },
    {
        'filename': 'LBNL_2026_United_States_Data_Center_Energy_Usage_Report_2025_Update.pdf',
        'title': 'United States Data Center Energy Usage Report: 2025 Update',
        'institution': 'Lawrence Berkeley National Laboratory / U.S. Department of Energy',
        'publication_date': '2026-06-18',
        'document_type': 'Official laboratory report PDF',
        'official_landing_url': 'https://escholarship.org/uc/item/33m6w3x0',
        'official_source_url': 'https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf',
        'retrieval_url': 'https://web.archive.org/web/20260726103937id_/https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf',
        'doi': '10.71468/P1RP4F',
        'license_note': 'Creative Commons Attribution 4.0, as stated on the eScholarship cover page.',
        'role_in_model': 'Independent U.S. bottom-up model using planned IT-equipment shipments, per-device power, utilization, and cooling simulations.',
        'representation': 'Complete official eScholarship PDF, retrieved from the Internet Archive snapshot dated 2026-07-26 because the live endpoint applies an automated-browser challenge',
        'expected_pages_min': 36,
        'expected_pages_max': 36,
        'title_checks': ['United States Data Center Energy Usage Report: 2025 Update', 'Lawrence Berkeley National Laboratory'],
    },
    {
        'filename': 'LBNL_2024_United_States_Data_Center_Energy_Usage_Report.pdf',
        'title': '2024 United States Data Center Energy Usage Report',
        'institution': 'Lawrence Berkeley National Laboratory / U.S. Department of Energy',
        'publication_date': '2024-12-19',
        'document_type': 'Official laboratory report PDF',
        'official_landing_url': 'https://eta-publications.lbl.gov/publications/2024-united-states-data-center',
        'official_source_url': 'https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report_1.pdf',
        'retrieval_url': 'https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report_1.pdf',
        'doi': '10.71468/P1WC7Q',
        'license_note': 'Consult the report and Berkeley Lab repository record for applicable reuse terms.',
        'role_in_model': 'Methodological predecessor and historical baseline for the 2026 Berkeley Lab update.',
        'representation': 'Complete original issuer-hosted PDF',
        'expected_pages_min': 79,
        'expected_pages_max': 79,
        'title_checks': ['United States Data Center Energy Usage Report', 'Lawrence Berkeley National Laboratory'],
    },
    {
        'filename': 'Federal_Reserve_2026_Monetary_Policy_Report_July.pdf',
        'title': 'Monetary Policy Report, July 2026',
        'institution': 'Board of Governors of the Federal Reserve System',
        'publication_date': '2026-07-10',
        'document_type': 'Official government report PDF',
        'official_landing_url': 'https://www.federalreserve.gov/monetarypolicy/2026-07-mpr-summary.htm',
        'official_source_url': 'https://www.federalreserve.gov/monetarypolicy/files/20260710_mprfullreport.pdf',
        'retrieval_url': 'https://www.federalreserve.gov/monetarypolicy/files/20260710_mprfullreport.pdf',
        'doi': '',
        'license_note': 'Official Federal Reserve publication; consult Federal Reserve website policies for reuse.',
        'role_in_model': 'Macroeconomic cross-check for U.S. business fixed investment and AI/data-center infrastructure spending.',
        'representation': 'Complete original issuer-hosted PDF',
        'expected_pages_min': 77,
        'expected_pages_max': 77,
        'title_checks': ['Monetary Policy Report', 'Board of Governors'],
    },
    {
        'filename': 'Federal_Reserve_2026_The_Global_Trade_Effects_of_the_AI_Infrastructure_Boom.html',
        'title': 'The Global Trade Effects of the AI Infrastructure Boom',
        'institution': 'Board of Governors of the Federal Reserve System — FEDS Notes',
        'publication_date': '2026-02-13',
        'document_type': 'Official HTML research note',
        'official_landing_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'official_source_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'retrieval_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'doi': '10.17016/2380-7172.3994',
        'license_note': 'Official Federal Reserve web publication; consult Federal Reserve website policies for reuse.',
        'role_in_model': 'Public-sector analysis of the AI infrastructure buildout, project pipeline, imports, and global trade effects.',
        'representation': 'Complete official HTML source downloaded from the Federal Reserve website',
        'expected_pages_min': None,
        'expected_pages_max': None,
        'title_checks': ['The Global Trade Effects of the AI Infrastructure Boom'],
    },
    {
        'filename': 'Federal_Reserve_2026_The_Global_Trade_Effects_of_the_AI_Infrastructure_Boom.pdf',
        'title': 'The Global Trade Effects of the AI Infrastructure Boom',
        'institution': 'Board of Governors of the Federal Reserve System — FEDS Notes',
        'publication_date': '2026-02-13',
        'document_type': 'Rendered PDF snapshot of official HTML research note',
        'official_landing_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'official_source_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'retrieval_url': 'https://www.federalreserve.gov/econres/notes/feds-notes/the-global-trade-effects-of-the-ai-infrastructure-boom-20260213.html',
        'doi': '10.17016/2380-7172.3994',
        'license_note': 'Local print-to-PDF representation of the official web page; the packaged HTML is the source-preserving representation.',
        'role_in_model': 'Portable visual snapshot of the Federal Reserve research note.',
        'representation': 'Locally rendered from the complete official live HTML page; not an issuer-supplied PDF',
        'expected_pages_min': 4,
        'expected_pages_max': 20,
        'title_checks': ['The Global Trade Effects of the AI Infrastructure Boom', 'Federal Reserve'],
    },
]

hash_to_name: dict[str, str] = {}
for item in sources:
    path = resources / item['filename']
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing or empty source file: {path}')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest in hash_to_name:
        raise RuntimeError(f'Duplicate source content: {path.name} and {hash_to_name[digest]}')
    hash_to_name[digest] = path.name

    item['retrieved_at_utc'] = retrieved
    item['file_size_bytes'] = path.stat().st_size
    item['sha256'] = digest

    if path.suffix.lower() == '.pdf':
        if not data.startswith(b'%PDF-'):
            raise RuntimeError(f'Invalid PDF header: {path}')
        reader = PdfReader(str(path), strict=False)
        pages = len(reader.pages)
        item['page_count'] = pages
        lo = item['expected_pages_min']
        hi = item['expected_pages_max']
        if lo is not None and not (lo <= pages <= hi):
            raise RuntimeError(f'Unexpected page count for {path.name}: {pages}; expected {lo}..{hi}')
        sample_text = '\n'.join((reader.pages[i].extract_text() or '') for i in range(min(4, pages)))
        normalized = ' '.join(sample_text.split()).lower()
        for phrase in item['title_checks']:
            if ' '.join(phrase.split()).lower() not in normalized:
                raise RuntimeError(f'Title/issuer verification failed for {path.name}: missing {phrase!r}')
    else:
        item['page_count'] = ''
        text = data.decode('utf-8', errors='replace')
        normalized = ' '.join(text.split()).lower()
        for phrase in item['title_checks']:
            if ' '.join(phrase.split()).lower() not in normalized:
                raise RuntimeError(f'HTML title verification failed for {path.name}: missing {phrase!r}')

# Preserve only public manifest fields.
public_fields = [
    'filename', 'title', 'institution', 'publication_date', 'document_type',
    'official_landing_url', 'official_source_url', 'retrieval_url', 'doi',
    'license_note', 'role_in_model', 'representation', 'retrieved_at_utc',
    'file_size_bytes', 'page_count', 'sha256'
]
public_sources = [{field: item.get(field, '') for field in public_fields} for item in sources]

with (root / 'SOURCE_MANIFEST.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=public_fields)
    writer.writeheader()
    writer.writerows(public_sources)

(root / 'SOURCE_MANIFEST.json').write_text(
    json.dumps(public_sources, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
)

checksum_lines = [f"{item['sha256']}  resources/{item['filename']}" for item in public_sources]
(root / 'SHA256SUMS.txt').write_text('\n'.join(checksum_lines) + '\n', encoding='utf-8')

readme = '''# AI Capex Macro Provenance Package

## Purpose

This package contains the official and government-sponsored source reports selected to ground a macro model of global AI and data-center capital spending. It intentionally excludes broker research, equity price targets, company-specific consensus forecasts, and promotional investment commentary.

## Directory structure

- `resources/` — complete reports and source representations with standardized filenames.
- `provenance/` — retrieval metadata for the archive-preserved Berkeley Lab PDF.
- `SOURCE_MANIFEST.csv` and `SOURCE_MANIFEST.json` — source identity, issuer, publication date, DOI, URLs, representation type, page count, byte size, modeling role, and SHA-256 hash.
- `SHA256SUMS.txt` — integrity checks for every packaged resource.
- `MODEL_SOURCE_MAP.md` — how each report is intended to support the macro model.

## Source hierarchy

1. **IEA 2026 — Key Questions on Energy and AI:** principal global update for data-center investment, electricity demand, and physical bottlenecks.
2. **IEA 2025 — Energy and AI:** foundational global physical and energy model.
3. **Berkeley Lab 2026:** independent U.S. bottom-up model based on equipment shipments, device power, utilization, and cooling.
4. **Berkeley Lab 2024:** methodological predecessor and historical baseline.
5. **Federal Reserve July 2026 Monetary Policy Report:** macroeconomic cross-check for U.S. fixed investment.
6. **Federal Reserve FEDS Note:** analysis of the international trade effects of the AI infrastructure buildout.

## Berkeley Lab 2026 retrieval note

The report's canonical source is the official eScholarship PDF. The live file endpoint applies an automated-browser challenge. The package therefore uses the Internet Archive's `20260726103937` snapshot of that exact official PDF, retrieved with the `id_` modifier so the original binary response body is returned without HTML rewriting. The corresponding CDX record is preserved in `provenance/LBNL_2026_Wayback_CDX_Record.json`.

## Methodological caution

Public-sector and international-agency sources are generally less exposed to stock-specific promotion than sell-side research, but they are not assumption-free. Forecasts remain scenario-conditioned estimates. The source manifest distinguishes original issuer PDFs, an archive-preserved official PDF, official HTML, and a locally rendered PDF snapshot.

## Integrity verification

From the package directory, run:

```bash
sha256sum -c SHA256SUMS.txt
```

Retrieval date: 2026-08-08 UTC.
'''
(root / 'README.md').write_text(readme, encoding='utf-8')

source_map = '''# Model Source Map

| Source | Primary modeling use | Variables supported |
|---|---|---|
| IEA 2026, *Key Questions on Energy and AI* | Global top-down capex and physical constraints | Cumulative data-center investment, electricity demand, energy-infrastructure share, bottlenecks, AI-focused capacity |
| IEA 2025, *Energy and AI* | Global base model and scenario architecture | Historical base, annual investment trajectory, data-center typology, power and energy scenarios |
| LBNL 2026, *United States Data Center Energy Usage Report: 2025 Update* | U.S. bottom-up compute deployment cross-check | Server and accelerator shipments, ASIC/GPU sensitivity, equipment lifetime, utilization, power draw, storage/network/facility load |
| LBNL 2024, *United States Data Center Energy Usage Report* | Historical and methodological baseline | Prior U.S. electricity-use range, installed-base method, server/storage/network assumptions |
| Federal Reserve, July 2026 *Monetary Policy Report* | Macroeconomic expenditure cross-check | U.S. business fixed investment, data-center construction, equipment/software investment |
| Federal Reserve FEDS Note, *The Global Trade Effects of the AI Infrastructure Boom* | International supply-chain and trade cross-check | AI infrastructure project pipeline, imported equipment, geographic production and trade exposure |

These sources do not by themselves determine company revenue or EPS. They establish the macro quantities and physical constraints from which supplier demand can subsequently be derived.
'''
(root / 'MODEL_SOURCE_MAP.md').write_text(source_map, encoding='utf-8')

notes = '''# Provenance Notes

## LBNL 2026 report

- Canonical official PDF: `https://escholarship.org/content/qt33m6w3x0/qt33m6w3x0.pdf`
- Official landing page: `https://escholarship.org/uc/item/33m6w3x0`
- DOI: `10.71468/P1RP4F`
- Internet Archive snapshot: `20260726103937`
- Wayback CDX digest recorded at retrieval: `EDMDZJGIFDMXRIUPSTE4V4YG2RGYADRY`
- CDX-reported MIME type: `application/pdf`
- CDX-reported status: `200`

The packaged PDF is verified as a 36-page PDF and its opening pages contain the official title, Lawrence Berkeley National Laboratory attribution, publication date, DOI, and Creative Commons Attribution notice.

## Federal Reserve FEDS Note

The Federal Reserve issued this item as a web publication rather than as a standalone PDF. The package includes both the downloaded official HTML source and a clearly labeled local print-to-PDF snapshot. The PDF snapshot is a convenience representation; the HTML file is the source-preserving copy.
'''
(root / 'PROVENANCE_NOTES.md').write_text(notes, encoding='utf-8')

print('Verified packaged resources:')
for item in public_sources:
    pages = f", {item['page_count']} pages" if item['page_count'] != '' else ''
    print(f"- {item['filename']}: {item['file_size_bytes']} bytes{pages}, sha256={item['sha256']}")
PY

# Additional structural checks where the runner image provides the utilities.
if command -v qpdf >/dev/null 2>&1; then
  while IFS= read -r -d '' pdf; do qpdf --check "$pdf"; done < <(find "$RES" -type f -name '*.pdf' -print0)
elif command -v pdfinfo >/dev/null 2>&1; then
  while IFS= read -r -d '' pdf; do pdfinfo "$pdf" >/dev/null; done < <(find "$RES" -type f -name '*.pdf' -print0)
fi

zip -r -9 "$OUTZIP" "$ROOT"
unzip -t "$OUTZIP"
test -s "$OUTZIP"
stat --printf='%n: %s bytes\n' "$OUTZIP"
