# GeoSentry

Unregulated mining and illegal logging go undetected for weeks because monitoring
relies on manual patrols and siloed satellite data — by the time anyone notices,
tens to hundreds of hectares are already gone. GeoSentry watches Sentinel-2, VIIRS,
and CHIRPS indicators continuously against a per-zone historical baseline, and stays
completely silent until a change is corroborated from multiple independent angles.
There's no dashboard to babysit and no review queue to triage — just one
plain-language alert, fired only when confidence clears a high bar.

The pipeline is deliberately adversarial with itself: an LLM proposes a cause
(mining, logging, agricultural expansion, drought, sensor artifact), a second LLM
pass argues the strongest alternative explanation, and rainfall + night-light data
either corroborate or contradict the story. All of it fuses into one posterior
confidence via a naive-Bayes update. Only above ~0.72 does GeoSentry speak. Everything
below that threshold is logged quietly, so the one alert that does arrive is one
worth trusting.

## Architecture

```
                         ┌─────────────────────────────┐
                         │   Google Earth Engine (GEE)  │
                         │  Sentinel-2 SR · Landsat 8/9 │
                         │  VIIRS DNB monthly · CHIRPS  │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ gee/                                                               │
 │  indices.py     → NDVI, BSI, cloud mask                            │
 │  baselines.py   → rolling median/MAD per zone, per calendar month  │
 │  detect.py      → z-score change detection  ──► ChangeIndicators   │
 │  corroborate.py → CHIRPS rainfall %ile, VIIRS night-light z-score  │
 └───────────────────────────────┬───────────────────────────────────┘
                                 │  indicators cross threshold
                                 ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ llm/ (Claude Sonnet 4.5)                                            │
 │  classify.py    → cause + confidence + reasoning                   │
 │  adversarial.py → devil's-advocate pass, argues an alternative     │
 └───────────────────────────────┬───────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ fusion/bayes.py                                                    │
 │  fuse_confidence(prior, evidence) → posterior via log-odds fusion  │
 │  evidence = {not_drought, nightlight_corroboration,                │
 │              adversarial_survives}                                 │
 └───────────────────────────────┬───────────────────────────────────┘
                                 │
                     posterior >= 0.72?
                     ┌───────────┴───────────┐
                    NO                       YES
                     │                        │
                     ▼                        ▼
           app/db.py: Detection      alerts/ntfy.py  → push notification
           (status="silent")         alerts/twilio_stub.py → logs SMS
           logged, no alert          app/db.py: Detection(status="alerted")
                                      + Alert row

 ┌───────────────────────────────────────────────────────────────────┐
 │ FastAPI (app/main.py)                                              │
 │  GET  /health  /zones  /detections  /alerts  /stats  /calibration  │
 │  POST /run-once?as_of=YYYY-MM-DD  → runs scripts/run_once.py       │
 └───────────────────────────────┬───────────────────────────────────┘
                                 ▼
                   web/index.html (vanilla JS, single file)
                   zone list · detections · confidence bars · alerts
```

## Quickstart

```bash
cd geosentry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # SYNTHETIC_MODE=true works with zero credentials

python scripts/seed_zones.py     # inserts 3 demo AOIs: mining, logging, control
python scripts/run_once.py       # runs the full pipeline once, prints results

python -m uvicorn app.main:app --reload   # then open http://127.0.0.1:8000
```

In the UI, pick a historical date and click **Run pipeline →** (or `POST
/run-once?as_of=YYYY-MM-DD`) to re-run the pipeline against real Earth Engine
imagery and watch a silent zone stay silent while a genuine anomaly clears the
confidence gate and pushes an ntfy alert.

## Demo features

- **Zone map** — a Leaflet map (CartoDB dark basemap) above the zone list, with
  one circle marker per zone colored by role (mining `#ff8c42`, logging
  `#f2c14e`, control `#7fb069`) and sized by the zone's last detected area
  (6-30px). Click a marker for cause, confidence, area, and timestamp. A "Fit
  all zones" button re-frames the map; markers update in place after **Fire
  demo alert** rather than rebuilding the map.
  *Screenshot: zone map with colored markers on a dark basemap.*

## Modes

- `SYNTHETIC_MODE=true` (default): `gee/*` and `llm/*` return deterministic fake
  data keyed off zone name/type, so the whole pipeline — including the "silent vs.
  alerted" split — is demoable offline with no API keys at all.
- `SYNTHETIC_MODE=false`: requires `GEE_SERVICE_ACCOUNT` +
  `GEE_SERVICE_ACCOUNT_KEY_PATH` and `ANTHROPIC_API_KEY`. Every module still falls
  back to synthetic behavior on any credential or API failure, so a flaky demo
  network never kills the presentation.
