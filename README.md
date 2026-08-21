# FPL expected-points model

A quantitative FPL model: pulls the live FPL API, estimates the probability
of every scoring event for every player, and solves for the optimal squad
as an integer program. Runs weekly on GitHub Actions.

## Hosting: you never run this yourself

Push this folder to a GitHub repo and the included Action does everything:
runs every Thursday 07:00 UTC on GitHub's servers, regenerates
`docs/index.html`, and commits it back. Turn on **Settings → Pages → Deploy
from branch → main → /docs** and the latest brief lives at
`https://<your-username>.github.io/<repo>/` — bookmark it on your phone.

Trigger an off-schedule run from **Actions → FPL weekly run → Run workflow**,
where you can pass your own squad ids and bank.

To run it locally instead:

```bash
pip install -r requirements.txt

# Optimal squad from scratch, 5-gameweek horizon
python -m fplmodel.run --horizon 5

# Transfer advice for a squad you already own
python -m fplmodel.run --squad 467,572,10,418,532,305,426,12,368,247,290,411,552,491,185 \
                       --bank 0.5 --free-transfers 1
```

Outputs land in `output/`: full projections CSV, optimal squad, ranked
transfers, and a markdown brief.

---

## How it works

### 1. Data

Two sources, both FPL API shaped, joined on `code` (the stable
cross-season player identifier — `id` is reassigned every summer).

| Source | Used for |
|---|---|
| `bootstrap-static` | Prices, ownership, availability, set-piece order, live season stats |
| `fixtures` | Schedule, home/away, blanks and doubles |
| [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) | Prior-season per-90 rates, for bootstrapping before the season has data |

No scraping. Nothing that breaks when a site changes its markup.

### 2. Team strength

Two independent estimates, blended:

- **xG-derived**: prior-season aggregated xG and minutes-weighted xGC per
  team, normalised to league average.
- **FPL ratings**: `strength_attack_*` / `strength_defence_*`, which cover
  all 20 clubs including promoted sides.

Non-promoted teams weight the xG estimate at 0.65. Promoted teams have no
prior-season data and fall back entirely to FPL ratings.

> **Caveat:** FPL zeroes out the granular attack/defence ratings until the
> season starts, so preseason runs fall back to the coarse 1–5 `strength`
> field. `_usable()` in `teams.py` handles the chain.

Fixture expectation for a given match:

```
xG_home = LEAGUE_GOALS × attack_home × opponent_leakiness_away × 1.05
xG_away = LEAGUE_GOALS × attack_away × opponent_leakiness_home × 0.95
```

The flat ±5% home adjustment is applied once at the end rather than baked
separately into xG and xGC.

### 3. Minutes

Nothing is assumed to start. Three quantities:

- **P(start)** — prior-season start rate, shrunk toward a low prior when
  the sample is thin (`λ = mins / (mins + 900)`), overridden by
  `chance_of_playing_next_round` and hard-zeroed on injury flags. Once the
  season is underway, current-season starts progressively take over.
- **P(60+ | start)** — logistic on mean minutes-per-start, centred at 62.
  A player averaging 64 min/start is far likelier to be hooked before the
  60-minute threshold than one averaging 88.
- **P(cameo)** — bench appearances, worth one point and little else.

### 4. Points

Every component is a probability times a value, summed:

| Component | Method |
|---|---|
| Appearance | `1 + P(60+)` |
| Goals | fixture-scaled xG/90 × position multiplier |
| Assists | fixture-scaled xA/90 × 3 |
| Clean sheet | `exp(−xGC_fixture) × P(60+) × position value` |
| Goals conceded | `E[floor(GC/2)]` over the Poisson, GK and DEF only |
| Saves | saves/90 ÷ 3, scaled by opponent attack |
| Defensive contributions | Poisson tail above the positional threshold |
| Bonus | prior realised bonus rate, fixture-scaled |

Two outputs per player: **`if_start`** (points given they start) and
**`true_total`** (weighted by the probability they play at all).

Excluded deliberately: red cards, own goals, penalty misses. Low-frequency
noise that distorts minutes-based estimates more than it predicts.

### 5. Optimisation

Binary integer program solved by CBC — squad membership, starting XI and
captain as linked decisions, subject to budget, 2/5/5/3 quotas, formation
validity, and max three per club. The full ~600-player pool, no
shortlisting: pre-filtering to a human candidate list reinjects exactly the
bias the model exists to remove.

Bench players carry a low weight rather than zero, so the solver still
prefers a bench that might actually play.

### 6. Sensitivity

For each strong player the solver rejected, the model **re-solves the
entire problem with that player forced in** and reports the total expected
value lost. That is the true reduced cost — unlike a raw score gap, it
accounts for the budget and club slots the forced pick consumes.

Reduced costs near zero mean the model is indifferent. That is the point at
which your read of the Friday press conference is worth more than the
output. Treat it as a licence to override, not a failure of the model.

---

## Where this differs from the video method

**Understat and the +40% assist boost are dropped.** The FPL API now
exposes `expected_goals` and `expected_assists` directly, Opta-sourced and
already aligned to FPL's own assist definition. The 40% correction exists
to reconcile Understat xA with FPL's more generous rules — applying it to
FPL's native xA would inflate every attacker for no reason.

**Transfermarkt is dropped.** `defensive_contribution_per_90` is a column.
Inferring CB-vs-LB to guess DefCon rates is unnecessary when the actual
DefCon rate is published, and it removes the most ToS-fragile scraper.

**Excel Solver is replaced by CBC.** Excel caps at 200 decision variables;
this problem has ~1,800 binaries before any multi-week transfer logic.
PuLP solves it exactly in under a second.

**Clean sheets use fixture-adjusted xGC, not the season average.**
`exp(−xGC)` is the right form, but feeding it a season-long mean ignores
who the opponent actually is.

Kept from the video: the minutes-probability approach, rolling long/short
form weighting, dropping red cards, the flat ±5% home adjustment, and the
sensitivity discipline.

---

## Known weaknesses

1. **Promoted sides are guesswork.** Hull, Ipswich and Coventry have no
   Premier League prior. They inherit FPL's coarse strength rating. The
   video's Championship→PL xG multipliers would improve this; nothing in
   the FPL API can.
2. **Bonus is modelled from realised history, not BPS mechanics.** Real BPS
   depends on in-match events and on who else played well. This is the
   weakest component.
3. **`DEFCON_THRESHOLD` in `config.py` is unverified.** Confirm against the
   current rules before trusting DefCon-driven picks.
4. **No rotation or European-fatigue term.** Teams in Europe rotate; the
   model does not know that.
5. **New signings have no prior data** and get shrunk to a low default.
   Expect the model to systematically undervalue them for ~6 gameweeks.
6. **`LEAGUE_GOALS_PER_TEAM` is hardcoded** at 1.42 rather than fitted.

## Calibration

Once gameweeks have been played, backtest against `data/<season>/gws/` in
the archive: correlate predicted `true_total` against realised points, and
tune `PRIOR_STRENGTH`, `XG_PRIOR_WEIGHT` and the logistic centre in
`_p60_from_mps`. Until then the parameters are reasoned defaults, not
fitted ones.
