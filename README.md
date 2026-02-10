# Reykjavik open accounting pipeline

This repo contains a small pipeline to download and prepare the City of Reykjavik open accounting (ársuppgjör) CSVs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download

```bash
python scripts/download_arsuppgjor.py
```

## Prepare

```bash
python scripts/prepare_arsuppgjor.py
```

Outputs:
- `data/raw/*.csv` (raw downloads)
- `data/raw/manifest.json` (download metadata)
- `data/processed/arsuppgjor_combined.parquet`
- `data/processed/arsuppgjor_metadata.json`

## Lookup entities (vm_number)

Resolve `vm_number` values via Skatturinn fyrirtækjaskrá and save names + URLs.

```bash
python scripts/lookup_vm_entities.py
```

Outputs:
- `data/processed/vm_entities.csv`
- `data/processed/vm_entities.json` (cache)

## Detect anomalies (CPI-adjusted)

Detect unusual increases/decreases in `actual` (`raun`) after CPI adjustment.

```bash
python scripts/detect_anomalies.py
```

Or:

```bash
make anomalies
```

Outputs:
- `data/processed/cpi_monthly.csv`
- `data/processed/cpi_annual.csv`
- `data/processed/anomalies_yoy_all.parquet`
- `data/processed/anomalies_yoy_all.csv`
- `data/processed/anomalies_flagged.parquet`
- `data/processed/anomalies_flagged.csv`
- `data/processed/anomalies_summary.json`

## Explore (Flask app)

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## Notes
- Column names are normalized to ASCII snake_case for easier exploration.
- The pipeline adds `source_file`, `source_url`, `year`, `period_month`, and `ingested_at` columns.
