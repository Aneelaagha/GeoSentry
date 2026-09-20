# GeoSentry

[Demo video](https://youtu.be/9tTvPc3gvN4?si=-lxZPbXzAxoOwxj8) · [GitHub](https://github.com/Aneelaagha/GeoSentry)

Unregulated mining and illegal logging go undetected for weeks because monitoring relies on manual patrols and siloed satellite data. By the time anyone notices, tens to hundreds of hectares are already gone. GeoSentry watches Sentinel-2, VIIRS, and CHIRPS indicators continuously against a per-zone historical baseline, and stays completely silent until a change is corroborated from multiple independent angles. There's no dashboard to babysit and no review queue to triage. Just one plain-language alert, fired only when confidence clears a high bar.

The pipeline is deliberately adversarial with itself: an LLM proposes a cause (mining, logging, agricultural expansion, drought, sensor artifact), a second LLM pass argues the strongest alternative explanation, and rainfall plus night-light data either corroborate or contradict the story. All of it fuses into one posterior confidence via a naive-Bayes update. Only above ~0.72 does GeoSentry speak. Everything below that threshold is logged quietly, so the one alert that does arrive is one worth trusting.

Every zone also carries a Land Degradation Neutrality (LDN) score aligned to UN SDG Indicator 15.3.1 (a 0-100 composite of productivity trend, land-cover stability, and recent-detection history), plus an IPCC 2006-default carbon-loss estimate (tCO₂e) for every real detection, so a flagged event reads as an actual climate-relevant quantity, not just a z-score.

> **Example, from the actual build.** On 2024-09-15, the pipeline detected 1,835 hectares of canopy loss in Novo Progresso at −5.12σ NDVI. Claude classified the cause as agricultural expansion at 85% confidence, drawing on real-world knowledge of the region as a documented deforestation frontier. Rainfall data independently conflicted, and the fused posterior stayed below the 0.72 gate. **No alert fired.** That refusal is the product. Even an 85%-confident LLM classification isn't enough on its own.

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
 │ llm/ (Claude Sonnet 4.5)                                           │
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
 │  POST /run-once?as_of=YYYY-MM-DD      → real historical run        │
 │  POST /run-once?demo=yanomami-2023    → replays a documented event │
 └───────────────────────────────┬───────────────────────────────────┘
                                 ▼
                   web/index.html (vanilla JS, single file)
                   satellite map w/ LDN-tinted zone polygons · stats ·
                   calibration · alert log · detection stream · silent log
```

## What's real and what's simulated

With `SYNTHETIC_MODE=false` and a real `ANTHROPIC_API_KEY` (verified end-to-end; all four flagged calibration candidates were classified by real Claude, not the synthetic fallback), everything in the pipeline is real:

- Google Earth Engine authentication and computation
- Sentinel-2 SR Harmonized, VIIRS DNB monthly, and CHIRPS daily ingestion
- Baseline computation (pooled per-pixel median + MAD across the trailing 3 sampled years, per zone per calendar month)
- Z-score change detection against those baselines (3-sigma gate)
- LLM cause classification and adversarial verification via Claude Sonnet 4.5 (real API calls)
- Bayesian confidence fusion with magnitude-scaled likelihood ratios
- Carbon loss estimation using IPCC 2006 AGB defaults
- SQLite persistence, alerting templates

`SYNTHETIC_MODE=true` (the default, zero-credential path) swaps GEE/Claude calls for deterministic synthetic data keyed off zone name/type, so the same detect → classify → fuse → gate pipeline is demoable offline. With `SYNTHETIC_MODE=false`, that fallback is real but not uniform: an LLM API failure is caught and classified synthetically (`llm/classify.py`), while a live Earth Engine failure is not. It surfaces as a real error (`POST /run-once` returns `{"error": "gee_unavailable"}`) rather than being silently replaced with fabricated output.

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

In the UI, pick a historical date and click **Run pipeline →** (or `POST /run-once?as_of=YYYY-MM-DD`) to re-run the pipeline against real Earth Engine imagery and watch a silent zone stay silent while a genuine anomaly clears the confidence gate and pushes an ntfy alert.

## Demo features

The dashboard is a clean, white, NASA-Earth-Observatory-style layout (Inter + IBM Plex Mono, generous whitespace). Two-column grid: map left, Overview / Calibration / Alert Log right, detection stream full-width below. Stacks to one column under 1000px. Everything on the page is backed by the routes above; nothing is scripted or hardcoded UI state.

- **Zone map.** Esri World Imagery satellite basemap under a light wash, with each zone rendered as its *real* AOI polygon (`Zone.aoi_geojson`, the same geometry every GEE query uses, not a placeholder shape), tinted by that zone's live LDN score (red `<40` · amber `40-69` · green `≥70`, both border and fill), plus a small reticle-marked circle on the centroid sized by last detected area. Click any zone (polygon or marker) for cause, confidence, area, and timestamp.
  - **Zone labels.** A permanent `Name · LDN` pill next to each marker at world/regional zoom, auto-hidden past zoom 8 once the polygons themselves are legible.
  - **Amazon / Global view.** The map defaults to the Amazon basin (3 of 5 zones sit there); a single top-right pill toggles to a fit-all-zones global view and back, always labeled with the view a click switches *to*.
  - **Country-name hover.** Hovering the map shows the country under the cursor, resolved client-side via a real point-in-polygon lookup against public-domain Natural Earth boundaries (no API key, no per-move network call).

- **LDN score & carbon loss (SDG 15.3.1).** The Overview card's headline number is the live average LDN score across all zones, color-coded by the same red/amber/green thresholds as the map. Every real detection also carries an estimated CO₂-equivalent loss (IPCC 2006 AGB defaults by NDVI baseline), shown on its card and summed across every suppressed calibration candidate ("estimated avoided carbon loss").

- **Calibration card.** A real historical backtest (25 windows swept across all 5 zones, reclassified by live Claude calls, not the synthetic fallback), shown zone-name-first with a verdict chip. Top 4 flagged candidates by default, with a "view all" toggle for the rest.

- **Detection stream.** Filtered to real events (a real affected area or a 3-sigma indicator) rather than routine no-change passes, which collapse into a single muted "N additional no-change observations in silent log" link. Each card shows real before/after Sentinel-2 thumbnails, a confidence bar, and a mono metadata line (cause, driving z-score, area, CO₂e).

- **Silent log.** A slide-in panel (▸ *View silent log*) listing every detection that stayed silent, each with a real, computed reason (ambiguous LLM confidence, a conflicting rainfall signal, VIIRS unavailable that month, etc.), not a canned message.

- **Demo replay.** The header includes a "Demo: replay Yanomami 2023" button. It routes a documented March 2023 garimpo surge through the same fusion and alerting pipeline as a live detection. The detection values are seeded from the real event; every downstream step (fusion, gate, ntfy push) runs production code. This is the only path in the system that seeds inputs, and it exists to demonstrate a fired alert since none of the 25 historical windows cleared the gate.

## Modes

The submitted demo runs with `SYNTHETIC_MODE=false` against real Earth Engine and real Claude. The default in `.env.example` is `true` so that a fresh clone can run `python scripts/run_once.py` with zero credentials.

- `SYNTHETIC_MODE=true` (default): `gee/*` and `llm/*` return deterministic fake data keyed off zone name/type, so the whole pipeline (including the "silent vs. alerted" split) is demoable offline with no API keys at all.
- `SYNTHETIC_MODE=false`: requires `GOOGLE_APPLICATION_CREDENTIALS` (path to a GEE service-account key JSON) + `GEE_PROJECT`, and `ANTHROPIC_API_KEY` for real Claude calls. The LLM layer falls back to a deterministic synthetic classifier on API failure. The GEE layer does not. A credential or network failure in `gee/*` surfaces as an error rather than fabricated output.

## Project layout

```
geosentry/
  app/                 FastAPI app, SQLite models, request/response schemas
  gee/                 Google Earth Engine: indices, baselines, detection, corroboration
  llm/                 Claude classification + adversarial verification pass
  fusion/              Bayesian confidence fusion
  alerts/              ntfy push and Twilio stub
  analysis/            LDN scoring, IPCC 2006 carbon estimation
  scripts/             Seed zones, seed baselines, run once, build calibration
  web/                 Single-file vanilla-JS dashboard
  secrets/             GEE service-account JSON (gitignored)
  calibration.json     Historical backtest results (25 windows)
  requirements.txt
  .env.example
```

## Why the silence is the feature

Most environmental monitoring tools alert on any anomaly. GeoSentry waits for corroboration. Across 25 real historical windows in the Amazon, DRC, and Uganda, the pipeline flagged 4 physical events and fired 0 alerts. Each flagged event had a real z-score, a real affected area, and a real carbon estimate behind it. None of them cleared the 0.72 confidence gate, because at least one independent signal (rainfall, night-lights, or the LLM's own certainty) failed to corroborate.

The alternative would have been to lower the gate and ship a system that fires regularly. That system would look busier on camera. It would also be ignored within a week.

## What we'd do with more time

- Replace the LLM with a fine-tuned classifier once labelled data exists. The LLM is the right starting point because it works on day one without training data, but a purpose-built model trained on confirmed events would be more consistent across cases.
- Compute a real NDVI trend slope for the LDN score's productivity sub-indicator. Today it's a documented neutral 50 (see `analysis/ldn.py`) because the pipeline only ever pools multi-year baselines down to a single median+MAD, discarding the year-by-year series a real trend needs. Filling that in is a scoped GEE addition, not a redesign.
- Scale zone coverage beyond five. The pipeline handles any AOI; the current zones are the demo set, not the limit.
- Calibrate the confidence gate against ground truth from an actual enforcement agency or conservation partner.

## License

MIT
