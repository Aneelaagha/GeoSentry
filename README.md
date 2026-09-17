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

Every zone also carries a Land Degradation Neutrality (LDN) score aligned to UN SDG
Indicator 15.3.1 — a 0-100 composite of productivity trend, land-cover stability, and
recent-detection history — plus an IPCC 2006-default carbon-loss estimate (tCO₂e) for
every real detection, so a flagged event reads as an actual climate-relevant quantity,
not just a z-score.

## Architecture

```
                         ┌─────────────────────────────┐
                         │   Google Earth Engine (GEE)  │
                         │  Sentinel-2 SR Harmonized    │
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
 │  evidence = {llm_conf, viirs_z, rain_percentile, area_ha} -        │
 │  each key optional; a missing/None viirs_z (VIIRS DNB lags real    │
 │  time 1-2 months) just skips that likelihood ratio, never fakes it │
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
                                             │
                                             ▼
                                   analysis/ldn.py
                                   compute_ldn_score() - 0-100 composite, SDG 15.3.1
                                   estimate_carbon_loss_tco2e() - IPCC 2006 AGB defaults

 ┌───────────────────────────────────────────────────────────────────┐
 │ FastAPI (app/main.py)                                              │
 │  GET  /health  /zones  /detections  /alerts  /silent-log           │
 │  GET  /stats  /calibration  /ldn/{zone_id}  /ldn/summary           │
 │  POST /run-once?as_of=YYYY-MM-DD  → runs scripts/run_once.py       │
 └───────────────────────────────┬───────────────────────────────────┘
                                 ▼
                   web/index.html (vanilla JS, single file)
                   satellite map w/ LDN-tinted zone polygons · stats ·
                   calibration · alert log · detection stream · silent log
```

**Real-Claude example:** On 2024-09-15 in Novo Progresso, the pipeline detected 1,835 hectares
of canopy loss at -5.12 sigma NDVI. Claude classified the cause as agricultural expansion at 85%
confidence, drawing on real-world knowledge of the region as a documented deforestation frontier.
The rainfall data independently conflicted with that classification, and the fused posterior
remained below the 0.72 gate. The system stayed silent. This is the product: even an
85%-confident LLM classification is not sufficient on its own. Independent physical evidence
must corroborate it.

## What's real and what's simulated

With `SYNTHETIC_MODE=false` and a real `ANTHROPIC_API_KEY` (verified end-to-end - all four
flagged calibration candidates were classified by real Claude, not the synthetic fallback),
everything in the pipeline is real:

- Google Earth Engine authentication and computation
- Sentinel-2 SR Harmonized, VIIRS DNB monthly, and CHIRPS daily ingestion
- Baseline computation (pooled per-pixel median + MAD across the trailing 3 sampled years, per
  zone per calendar month)
- Z-score change detection against those baselines (3-sigma gate)
- LLM cause classification and adversarial verification via Claude Sonnet 4.5 (real API calls)
- Bayesian confidence fusion with magnitude-scaled likelihood ratios
- Carbon loss estimation using IPCC 2006 AGB defaults
- SQLite persistence, alerting templates

`SYNTHETIC_MODE=true` (the default, zero-credential path) swaps GEE/Claude calls for
deterministic synthetic data keyed off zone name/type, so the same detect → classify → fuse →
gate pipeline is demoable offline. Every real-path module also falls back to synthetic on any
credential or API failure, so a flaky network never kills a live demo.

## Quickstart

```bash
cd geosentry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # SYNTHETIC_MODE=true works with zero credentials

python scripts/seed_zones.py     # inserts 5 demo AOIs: 2 mining, 2 logging, 1 control
python scripts/run_once.py       # runs the full pipeline once, prints results

python -m uvicorn app.main:app --reload   # then open http://127.0.0.1:8000
```

In the UI, pick a historical date and click **Run pipeline →** (or `POST
/run-once?as_of=YYYY-MM-DD`) to re-run the pipeline against real Earth Engine
imagery and watch a silent zone stay silent while a genuine anomaly clears the
confidence gate and pushes an ntfy alert.

## Demo features

The dashboard is a clean, white, NASA-Earth-Observatory-style layout (Inter +
IBM Plex Mono, generous whitespace) — a two-column grid (map left, Overview /
Calibration / Alert Log right, detection stream full-width below) that stacks
to one column under 1000px. Everything on the page is backed by the routes
above; nothing is scripted or hardcoded UI state.

- **Zone map** — Esri World Imagery satellite basemap under a light wash, with
  each zone rendered as its *real* AOI polygon (`Zone.aoi_geojson`, the same
  geometry every GEE query uses — not a placeholder shape), tinted by that
  zone's live LDN score (red `<40` · amber `40-69` · green `≥70`, both border
  and fill), plus a small reticle-marked circle on the centroid sized by last
  detected area. Click any zone (polygon or marker) for cause, confidence,
  area, and timestamp.
  - **Zone labels** — a permanent `Name · LDN` pill next to each marker at
    world/regional zoom, auto-hidden past zoom 8 once the polygons themselves
    are legible.
  - **Amazon / Global view** — the map defaults to the Amazon basin (3 of 5
    zones sit there); a single top-right pill toggles to a fit-all-zones
    global view and back, always labeled with the view a click switches *to*.
  - **Country-name hover** — hovering the map shows the country under the
    cursor, resolved client-side via a real point-in-polygon lookup against
    public-domain Natural Earth boundaries (no API key, no per-move network
    call).
  *Screenshot: satellite map with LDN-tinted zone polygons and the Amazon-view
  legend.*

- **LDN score & carbon loss (SDG 15.3.1)** — the Overview card's headline
  number is the live average LDN score across all zones, color-coded by the
  same red/amber/green thresholds as the map. Every real detection also
  carries an estimated CO₂-equivalent loss (IPCC 2006 AGB defaults by NDVI
  baseline), shown on its card and summed across every suppressed calibration
  candidate ("estimated avoided carbon loss").

- **Calibration card** — a real historical backtest (25 windows swept across
  all 5 zones, reclassified by live Claude calls, not the synthetic
  fallback), shown zone-name-first with a verdict chip; the top 4 flagged
  candidates by default, with a "view all" toggle for the rest.

- **Detection stream** — filtered to real events (a real affected area or a
  3-sigma indicator) rather than routine no-change passes, which collapse
  into a single muted "N additional no-change observations in silent log"
  link. Each card shows real before/after Sentinel-2 thumbnails, a confidence
  bar, and a mono metadata line (cause, driving z-score, area, CO₂e).

- **Silent log** — a slide-in panel (▸ *View silent log*) listing every
  detection that stayed silent, each with a real, computed reason (ambiguous
  LLM confidence, a conflicting rainfall signal, VIIRS unavailable that
  month, etc.) — not a canned message.

## Modes

- `SYNTHETIC_MODE=true` (default): `gee/*` and `llm/*` return deterministic fake
  data keyed off zone name/type, so the whole pipeline — including the "silent vs.
  alerted" split — is demoable offline with no API keys at all.
- `SYNTHETIC_MODE=false`: requires `GOOGLE_APPLICATION_CREDENTIALS` (path to a GEE
  service-account key JSON) + `GEE_PROJECT`, and `ANTHROPIC_API_KEY` for real Claude
  calls (falls back to a deterministic synthetic classifier if unset). Every module
  still falls back to synthetic behavior on any credential or API failure, so a
  flaky demo network never kills the presentation.

## What we'd do with more time

- Replace the LLM with a fine-tuned classifier once labelled data exists. The LLM is the
  right starting point because it works on day one without training data, but a
  purpose-built model trained on confirmed events would be more consistent across cases.
- Compute a real NDVI trend slope for the LDN score's productivity sub-indicator. Today
  it's a documented neutral 50 (see `analysis/ldn.py`) because the pipeline only ever
  pools multi-year baselines down to a single median+MAD, discarding the year-by-year
  series a real trend needs — filling that in is a scoped GEE addition, not a redesign.
