# bagua-weather

An 8-state Markov regime model of weather for energy risk, built on open [DWD](https://opendata.dwd.de/) data (station 01443, Freiburg).

**[→ Source code on GitHub](https://github.com/shaoyanji/bagua-weather)** · MIT license

## What it does

Classifies hourly weather into an 8-state regime space (`sun × wind × wet`), masks night hours, fits a Markov transition matrix on a chronological train split, and scores held-out next-state prediction. Ships `bagua fetch | fit | eval | ablate`.

## Held-out results (2016-01-01 split, station 01443)

| Model | Log loss ↓ | Top-1 ↑ |
|---|---|---|
| climatology | 1.388 | 0.444 |
| persistence | 1.048 | 0.750 |
| markov_global | 0.784 | 0.750 |
| markov_contextual | 0.769 | 0.750 |
| gbm (tuned, optional sklearn) | **0.758** | 0.750 |

Stationary-vs-empirical TVD: **0.037** — the fitted chain is well calibrated long-run.

## State-count ablation (`bagua ablate`)

Cross-comparison — richer chains marginalized onto coarser targets, scored on identical test rows:

| Target | Native LL | Via 8-state | Via 16-state |
|---|---|---|---|
| sun×wet | 0.571 | 0.564 | 0.562 |
| sun×wind | 0.615 | 0.605 | 0.602 |
| wind×wet | 0.425 | 0.409 | 0.407 |
| sun×wind×wet | 0.784 | — | **0.780** |

**Verdict: 8 states is justified.** Every bit earns its place; the 16-state chain's apparent gain is mostly that temperature is a richer target — it buys only −0.004 LL on the sun/wind/wet question.

## Honest limitations

- Dwell times >24h are structurally zero — night masking truncates every run at nightfall (right-censored)
- Threshold sensitivity grid not yet run
- Temperature bit only pays off if a downstream payoff needs it (heating/cooling demand)

*Generated from the v0.1.0 results; see the [README](https://github.com/shaoyanji/bagua-weather#readme) for the full spec and method.*
