# bagua-weather

**An 8-state Markov regime model of weather for energy risk, built on open DWD data.**

Status: **v0.1.0** — M1–M4 done on station 01443 (Freiburg). Numbers below are from real runs; see §12.

## 1. Idea

Weather is a discrete-state process at the level that matters for energy trading: is it sunny, windy, wet? Three binary signals give 2^3 = 8 regimes. This project:

1. Discretises hourly DWD observations into 8 states (named after the bagua trigrams as a labelling/visual layer).
2. Fits a first-order Markov chain (global and season × time-of-day conditioned).
3. Tests, honestly, whether this beats simple baselines on held-out years.
4. (v0.2.0) Uses the chain to simulate scenario paths for pricing PPA / weather-derivative payoffs.

The trigram framing is a naming layer only. **The numbers have to carry the project.** If 8 states are not justified against 4 or 16, report that.

## 2. Scope

**v0.1.0 (this spec)**
- Ingest hourly sunshine, wind and precipitation for one station (Freiburg, DWD station id 01443, verify).
- Classify hourly states, mask night hours.
- Fit global and contextual Markov models; evaluate on a chronological hold-out.
- Compare against baselines; write a reproducible report with figures.

**Out of scope for v0.1.0:** simulation/pricing (v0.2.0), multi-station, forecasting beyond one step, a web UI.

## 3. Data

Source: DWD Climate Data Center, open hourly station observations
(`https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/`).

| Variable | DWD product dir | Column | Unit | Notes |
|---|---|---|---|---|
| Sunshine duration | `sun` | `SD_SO` | min per hour | 0 at night by construction |
| Wind speed | `wind` | `F` | m/s | hourly mean (direction is `D`) |
| Precipitation | `precipitation` | `R1` | mm | `RS_IND` is the 0/1 indicator |

Verified against the DESCRIPTION PDFs and the station-01443 metadata (2026-11-19): the column table above is correct. **Timestamp caveat:** `MESS_DATUM` (`YYYYMMDDHH`) is *not* uniformly UTC — Freiburg sun data is WOZ before 2002-07-01 and wind is MEZ before 2003-09-01; the all-UTC overlap begins **2003-09-01**, which `load_hourly` applies as the default `since` cutoff.

Requirements and traps:
- **Discover files by listing the directory**; do not hard-code zip names. Both `historical` and `recent` folders exist and must be merged (de-duplicate overlapping timestamps).
- **Verify column names, units and timestamp convention (UTC vs local) against DWD's per-product description PDFs** before trusting anything above. I wrote this table from memory.
- DWD uses `-999` as a missing-value sentinel. Convert to NaN.
- Cache downloads in `data/raw/` (gitignored); ingestion must be idempotent.
- Be polite to the server: no parallel hammering, reuse the cache.

## 4. State definition

State = `sun*4 + wind*2 + wet` (bit order bottom→top = sun, wind, wet).

| Bit | Rule (defaults, configurable) |
|---|---|
| sun | `sun_minutes >= 30` |
| wind | `wind_ms >= 5.0` |
| wet | `precip_mm >= 0.1` |

Trigram labels use the Fu Xi order: 7 = ☰ Qian … 0 = ☷ Kun. See `states.py`.

**Night handling (important):** the sun bit is always 0 at night, which would fake persistence and inflate every metric. Use `solar.mask_night` (pvlib solar elevation > 5°) to set night states to NA. Transitions are only counted between *consecutive daylight hours*. Document the resulting data coverage in the report.

## 5. Model

- **Global:** 8×8 row-stochastic matrix, Laplace smoothing `alpha`.
- **Contextual:** one matrix per season (DJF/MAM/JJA/SON) × 6-hour UTC bucket, falling back to global for unseen contexts.
- Utilities: stationary distribution, path sampling.
- Planned: second-order chain, HMM baseline (e.g. `hmmlearn`, optional dependency).

## 6. Evaluation

Chronological split (default: train < 2016-01-01, test ≥ 2016-01-01; keep configurable). Never shuffle.

Models compared on next-state prediction:

| Model | Description |
|---|---|
| `climatology` | marginal distribution, ignores previous state |
| `persistence` | stay with empirical stay-rate, else uniform (smoothed) |
| `markov_global` | first-order chain |
| `markov_contextual` | chain per season × hour bucket |
| *(M4)* HMM or gradient boosting | a stronger baseline |

Metrics: log loss (primary), top-1 accuracy, plus
- stationary distribution vs empirical frequencies,
- **dwell-time distribution** (a first-order chain gives geometric dwell times; check how badly real weather deviates and say so),
- **state-count ablation** (stretch): 4 states (drop one bit), 8, 16 (add a temperature bit from `TT_TU`), compared on held-out likelihood per state-count-adjusted terms. Be careful comparing log loss across different state counts; describe the method used.

## 7. Repo layout

```
src/bagua_weather/
  states.py      # thresholds, encode/decode, trigram labels, classify()   [done]
  solar.py       # daylight masking via pvlib                              [done]
  model.py       # transition pairs, Markov fit, contextual, sampling      [done]
  baselines.py   # climatology, persistence, GBM (M4, sklearn extra)       [done M4]
  evaluate.py    # chronological eval vs baselines                         [done]
  ingest.py      # DWD fetch + load_hourly                                 [done M1]
  fit.py         # fit pipeline: classify->mask->pairs->fit->save          [done M2]
  report.py      # metrics.json + figures                                  [done M3]
  simulate.py    # scenario paths + payoff pricing                         [v0.2.0]
  cli.py         # `bagua fetch|fit|eval`                                  [done M2 fetch/fit]
tests/           # pytest; synthetic-chain tests must keep passing
data/            # raw/ and processed/ are gitignored
reports/         # generated report + figures (gitignored images)
```

## 8. CLI (target)

```
bagua fetch --station 01443          # download + cache
bagua fit   --station 01443          # classify, mask night, fit, save matrices to data/processed/
bagua eval  --station 01443 --split 2016-01-01   # metrics JSON + figures in reports/
```

## 9. Milestones and acceptance criteria

- **M1 Ingest.** `fetch_station` + `load_hourly` return a UTC-indexed hourly frame for station 01443 with `sun_minutes, wind_ms, precip_mm`, -999 → NaN, historical+recent merged. *Accept:* unit test with a small fixture file; no network in tests.
- **M2 Fit.** `bagua fit` runs end to end: classify → mask night → fit → save matrices. *Accept:* matrices are row-stochastic; report logs data coverage after night/missing masking.
- **M3 Evaluate.** `bagua eval` writes `reports/metrics.json` and figures: transition-matrix heatmap with trigram labels, state frequencies, dwell-time histogram vs geometric fit. *Accept:* Markov models are reported against both baselines whatever the outcome.
- **M4 Stronger baseline.** Add HMM or gradient boosting; compare. *Accept:* result written up honestly in the report.
- **M5 (stretch) Ablation.** 4 / 8 / 16 states.
- **v0.1.0 release** = M1 to M3 done, M4 attempted, README results section filled with real numbers.

## 10. Risks / honest caveats

- First-order Markov dynamics may be too weak; the dwell-time check will show it. A negative or modest result is still a publishable finding for the portfolio.
- Threshold choices are arbitrary; report sensitivity (a small grid) rather than one lucky setting.
- One station is not a region. Say so.
- Do not present trigram symbolism as a modelling result.

## 11. Development

```
nix develop            # or: python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## 12. Results

Real run, station 01443 (Freiburg), `bagua eval` at split 2016-01-01.
Data: 203,297 UTC-hourly rows (2003-09-01 → 2026-11-18) → 152,026 classified states
→ **91,630 daylight states** after the solar-elevation mask → 44,137 train / 38,983
test transition pairs.

**Next-state prediction (held-out, daylight hours only):**

| Model | Log loss ↓ | Top-1 acc ↑ |
|---|---|---|
| climatology | 1.388 | 0.444 |
| persistence | 1.048 | 0.750 |
| markov_global | 0.784 | 0.750 |
| markov_contextual | 0.769 | 0.750 |
| **gbm (M4)** | **0.758** | **0.750** |

Findings:

- All structured models beat climatology decisively; the Markov chain beats persistence on *probability quality* (log loss) at equal top-1 — it knows *how sure* to be, not just *what* is likeliest.
- Context (season × hour bucket) adds only ~0.015 log loss — Freiburg's hourly regimes are already well captured by the previous state alone.
- The tuned GBM baseline (categorical `prev`, l2, early stopping on train-internal validation) edges the contextual chain (0.758 vs 0.769) — a narrow win, not a rout. The 8-state first-order model is competitive.
- Stationary distribution vs empirical test frequencies: TVD **0.037** — the chain's long-run state mix matches reality well.
- Dwell times: empirical mean runs are *shorter* than the geometric prediction (e.g. Zhen ☳ 4.8h vs 7.1h, Kun ☷ 2.7h vs 3.5h). Partly this is censoring — the daylight mask truncates every run at nightfall, so >24h runs are structurally impossible — but short-run deviations are also visible within daylight windows.

**Limitations (honest):**

- One station, one era. Freiburg 2003–2026 says nothing about coastal or alpine regimes.
- Dwell comparison is right-censored by the daylight mask; cross-midnight regime persistence is unmeasured by construction.
- GBM early stopping uses a *random* 15% of the train slice — not chronological — acceptable inside train, stated anyway.
- Sun-column gaps: `sun_minutes` is missing for ~25% of rows (instrument outages), so "daylight states" understates true coverage.
- M5 state-count ablation (4/8/16 states) not run — stretch goal, open.
- Thresholds (sun 30min, wind 5m/s, wet 0.1mm) are the spec defaults; no sensitivity grid run yet.

## License

TBD (suggest MIT).
