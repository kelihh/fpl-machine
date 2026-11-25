# Calibration, measured

Run 2026-11-25, after GW11. Out-of-sample via `fplmodel.backtest`, which
neutralises today's availability flags and recomputes per-90 rates as of
the gameweek before, so each projection uses only pre-deadline data.

## 1. Player ranking: real skill, beats both free baselines

Full pool, ~460 players per gameweek, GW3-11:

| | mean |
|---|---|
| model Pearson r | **0.424** |
| model Spearman rho | **0.495** |
| ownership baseline r | 0.329 |
| price baseline r | 0.194 |

Beats ownership in 7 of 9 weeks and price in 9 of 9. The README's
blended-vs-unblended figures (0.430 vs 0.370) hold up.

## 2. But the skill is all between tiers, not within them

Spearman rho restricted to narrow prediction bands:

| gw | full pool | predicted 3.0-7.5 | top 60 |
|---|---|---|---|
| 3 | 0.534 | 0.133 | 0.292 |
| 4 | 0.498 | 0.195 | 0.143 |
| 5 | 0.515 | 0.115 | 0.151 |
| 6 | 0.434 | 0.110 | 0.190 |
| 7 | 0.518 | -0.109 | 0.030 |
| 8 | 0.456 | -0.063 | -0.082 |
| 9 | 0.516 | 0.140 | 0.273 |
| 10 | 0.513 | 0.340 | 0.118 |
| 11 | 0.475 | 0.119 | 0.015 |
| **mean** | **0.495** | **0.109** | **0.126** |

Within the 3.0-7.5 band the model is close to blind, and negative in two
of nine weeks. That band is where *every decision inside a 15-man squad
lives*: bench order, the third-through-fifth midfielder, one 6.0m
defender against another, and most captaincy choices.

**Use it for transfers, where it selects from the full pool and the
0.495 is real. Do not trust it to order a bench.** A gap of ~0.5 EP
between two players in the same squad is noise. A gap of 4+, like a
non-playing reserve keeper against a starter, is signal.

## 3. Squad-level totals run hot, and the intervals are too narrow

Kelih's actual gameweeks against what was projected for the XI he fielded:

| gw | projected | actual | ratio | inside p10-p90? |
|---|---|---|---|---|
| 6 | 64.3 | 39.0 | 0.61 | no |
| 9 | 63.0 | 46.0 | 0.73 | yes |
| 10 | 68.6 | 52.0 | 0.76 | yes |
| 11 | 73.4 | 43.0 | 0.59 | no |

Four undershoots in four, mean ratio 0.67, and two outside an interval
built to miss one in five. So the *level* is biased high and the
simulated spread is too tight, even though the *ranking* is sound.

Not corrected by a fitted fudge factor: n=4 is too thin, and scaling the
output would hide the cause. Prime suspects remain the three the README
already names -- bonus modelled from realised history rather than BPS
mechanics, no rotation or European-fatigue term, and an unverified
`DEFCON_THRESHOLD`. Actuals above are inferred from per-element
`event_points` for the XI believed fielded, not read from the entry.
