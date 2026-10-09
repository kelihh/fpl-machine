"""Correlated Monte Carlo over a starting XI.

Expected points are additive and therefore blind to the thing that
actually decides a gameweek: an XI with three players in one match does
not have three independent chances of a good score, it has one. Here the
*match* is simulated first -- one scoreline per fixture -- and players are
drawn conditional on it, so teammates share a clean sheet and share a
scoreline.

Crucially this also prices the opposite case. Two players on opposing
sides of the same fixture are *negatively* correlated: a goal for one is a
goal against the other. A naive "max N players per fixture" rule treats
that hedge as if it were concentration, which is backwards. The quantity
to cap is players per *side*, and the simulator is what shows the
difference.

What this is for is the shape of the distribution -- P(disaster),
P(big week) -- not a better mean. The mean barely moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def simulate_xi(
    detail: pd.DataFrame, xi: list[int], captain: int,
    n_sims: int = 40_000, seed: int = 7, triple: bool = False,
) -> tuple[np.ndarray, dict]:
    """Total points for the XI across `n_sims` simulated gameweeks."""
    rng = np.random.default_rng(seed)
    d = detail[detail["id"].isin(xi)].drop_duplicates("id").set_index("id").loc[xi]

    # ---- one scoreline per fixture, shared by both sides
    scoreline = {}
    for fid, g in d.groupby("fixture"):
        r = g.iloc[0]
        if bool(r["is_home"]):
            lam_h, lam_a = float(r["xg_for"]), float(r["xg_against"])
        else:
            lam_h, lam_a = float(r["xg_against"]), float(r["xg_for"])
        scoreline[fid] = (rng.poisson(lam_h, n_sims), rng.poisson(lam_a, n_sims))

    totals = np.zeros(n_sims)
    per_player = {}

    # Attacking share: a team's goals are handed out in proportion to each
    # player's xG share of that team, with a residual for everyone not in
    # the XI (so two teammates can both blank on a 3-0 win).
    for (fid, team), g in d.groupby(["fixture", "team"]):
        G_h, G_a = scoreline[fid]
        home = bool(g["is_home"].iloc[0])
        gf, ga = (G_h, G_a) if home else (G_a, G_h)

        team_xg = max(float(g["xg_for"].iloc[0]), 0.05)
        share_g = (g["sim_xg"].to_numpy() / team_xg).clip(0, 0.95)
        share_a = (g["sim_xa"].to_numpy() / team_xg).clip(0, 0.95)

        for k, pid in enumerate(g.index):
            r = g.loc[pid]
            pos = r["pos"]
            start = rng.random(n_sims) < float(r["p_start"])
            long = start & (rng.random(n_sims) < float(r["p60_given_start"]))
            cameo = (~start) & (rng.random(n_sims) < float(r["p_cameo"]))
            on = start | cameo

            pts = np.where(long, 2.0, np.where(on, 1.0, 0.0))

            goals = rng.binomial(gf, share_g[k]) * on
            assists = rng.binomial(gf, share_a[k]) * on
            pts += goals * C.GOAL_POINTS[pos] + assists * C.ASSIST_POINTS

            if pos in ("GK", "DEF", "MID"):
                pts += (ga == 0) * long * C.CLEAN_SHEET_POINTS[pos]
            if pos in ("GK", "DEF"):
                pts -= np.floor(ga / 2) * long

            if pos == "GK":
                rate = float(r["p_saves90"]) * float(r["exp_mins_if_start"]) / 90.0
                pts += np.floor(rng.poisson(max(rate, 0.1), n_sims) / 3.0) * long

            dc_lam = max(float(r["p_dc90"]) * float(r["exp_mins_if_start"]) / 90.0, 0.01)
            pts += (rng.poisson(dc_lam, n_sims) >= C.DEFCON_THRESHOLD[pos]) * C.DEFCON_POINTS * on

            # Bonus: a baseline rate, lifted when the player actually returned.
            bon = rng.poisson(max(float(r["p_bonus90"]) * 0.8, 0.02), n_sims)
            bon = np.minimum(bon + (goals > 0) * rng.integers(1, 4, n_sims), 3)
            pts += bon * on

            mult = (3.0 if triple else 2.0) if pid == captain else 1.0
            per_player[pid] = pts
            totals += pts * mult

    return totals, per_player


def summarise(totals: np.ndarray, label: str) -> dict:
    return {
        "squad": label,
        "mean": round(float(totals.mean()), 1),
        "sd": round(float(totals.std()), 1),
        "p10": int(np.percentile(totals, 10)),
        "median": int(np.percentile(totals, 50)),
        "p90": int(np.percentile(totals, 90)),
        "P<40": round(float((totals < 40).mean()) * 100, 1),
        "P>70": round(float((totals > 70).mean()) * 100, 1),
    }
