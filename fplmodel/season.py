"""In-season data: current-season rates and a team-form overlay.

The prior-season rates in `data.py` are the only thing the model has in
August. By October they are the *minority* of the available evidence and
holding onto them unchanged is the single biggest avoidable error in the
engine.

Two corrections live here:

  blend_rates   -- per-90 rates become a minutes-weighted blend of this
                   season and last. A player with 450 minutes on the board
                   has said something; a player with 30 has not.
  form_overlay  -- team attack/defence multipliers get pulled toward the
                   goals actually scored and conceded so far.

Both weights are fitted by backtest (see `backtest.py`), not guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

# Per-90 rate columns, and the current-season counters they come from.
RATE_MAP = {
    "p_xg90": "expected_goals",
    "p_xa90": "expected_assists",
    "p_xgc90": "expected_goals_conceded",
    "p_dc90": "defensive_contribution",
    "p_saves90": "saves",
    "p_bonus90": "bonus",
    "p_gc90": "goals_conceded",
}


def fetch_history(path=None, max_age_hours: float = 6.0) -> pd.DataFrame:
    """Per-gameweek history for every player with minutes, from the API.

    `bootstrap-static` gives season totals but not the per-gameweek splits,
    and the splits are what `blend_rates` and the backtest both need. One
    `element-summary` call per player, which is ~450 requests and finishes in
    a couple of seconds in parallel. Cached so repeated local runs are free;
    in CI the cache is always cold, which is the case this function exists
    for -- before it, the scheduled run silently fell back to prior-season
    rates and published a different model from the one being tuned.
    """
    import concurrent.futures as cf
    import time

    path = Path(path or (C.CACHE / "gw_history.csv"))
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return load_history(path)

    bs = json.loads((C.CACHE / "bootstrap.json").read_text())
    ids = [int(e["id"]) for e in bs["elements"]
           if pd.to_numeric(e.get("minutes"), errors="coerce") > 0]

    session = requests.Session()
    session.headers.update({"User-Agent": "fpl-model/1.0"})

    def one(pid: int):
        for _ in range(3):
            try:
                r = session.get(C.FPL_ELEMENT_SUMMARY.format(pid), timeout=25)
                if r.status_code == 200:
                    return r.json().get("history", [])
            except requests.RequestException:
                time.sleep(0.5)
        return []

    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        for hist in pool.map(one, ids):
            rows.extend(hist)

    if not rows:
        raise RuntimeError("element-summary returned nothing; refusing to "
                           "publish a run with no current-season data")
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return load_history(path)


def load_history(path=None) -> pd.DataFrame:
    """Per-player, per-gameweek history for the current season."""
    path = path or (C.CACHE / "gw_history.csv")
    h = pd.read_csv(path)
    for c in h.columns:
        if c == "kickoff_time":
            continue
        conv = pd.to_numeric(h[c], errors="coerce")
        if conv.notna().any():
            h[c] = conv
    return h


def current_aggregates(hist: pd.DataFrame, as_of_gw: int) -> pd.DataFrame:
    """Cumulative current-season totals through `as_of_gw`, per player."""
    h = hist[hist["round"] <= as_of_gw]
    cols = list(set(RATE_MAP.values())) + ["minutes", "starts", "total_points"]
    agg = h.groupby("element")[cols].sum()
    agg["games"] = h.groupby("element")["round"].nunique()
    agg["mins_per_start"] = agg["minutes"] / agg["starts"].replace(0, np.nan)
    return agg


def blend_rates(
    df: pd.DataFrame, hist: pd.DataFrame, as_of_gw: int,
    k_attack: float = C.BLEND_K_ATTACK,
    k_defence: float = C.BLEND_K_DEFENCE,
    rates: bool = True,
) -> pd.DataFrame:
    """Replace prior-season rates with a current/prior minutes-weighted blend.

    Weight on this season is  mins / (mins + k).  Attacking rates get a
    larger k because xG per 90 is noisier over a handful of matches than
    a counting stat like tackles+interceptions, which stabilises fast.
    """
    out = df.copy()
    agg = current_aggregates(hist, as_of_gw)
    cur_mins = out["id"].map(agg["minutes"]).fillna(0.0)

    for rate_col, src in (RATE_MAP.items() if rates else []):
        k = k_defence if rate_col in ("p_dc90", "p_saves90", "p_xgc90", "p_gc90") else k_attack
        w = (cur_mins / (cur_mins + k)).clip(0, 1)
        cur_rate = (out["id"].map(agg[src]) / cur_mins.replace(0, np.nan)) * 90.0
        cur_rate = pd.to_numeric(cur_rate, errors="coerce")
        prior = pd.to_numeric(out[rate_col], errors="coerce").fillna(0.0)
        # Where there is no current-season rate, keep the prior untouched.
        out[rate_col] = np.where(cur_rate.notna(), w * cur_rate + (1 - w) * prior, prior)

    # Minutes-per-start is about role, and role this season beats role last.
    cur_mps = out["id"].map(agg["mins_per_start"])
    w_mps = (cur_mins / (cur_mins + 180.0)).clip(0, 1)
    prior_mps = pd.to_numeric(out["prior_mins_per_start"], errors="coerce")
    out["prior_mins_per_start"] = np.where(
        cur_mps.notna(),
        w_mps * cur_mps + (1 - w_mps) * prior_mps.fillna(cur_mps),
        prior_mps,
    )

    # Overwrite the live cumulative counters so `minutes_model` sees only
    # what was known at `as_of_gw` -- essential for honest backtesting.
    out["minutes"] = cur_mins
    out["starts"] = out["id"].map(agg["starts"]).fillna(0.0)
    out["gw_played"] = as_of_gw
    return out


# ------------------------------------------------------------------ form
def form_overlay(
    strength: pd.DataFrame, fixtures: pd.DataFrame, as_of_gw: int,
    weight: float = C.FORM_WEIGHT,
) -> pd.DataFrame:
    """Pull team attack/defence multipliers toward results so far.

    Uses actual goals, not xG: FPL points are paid on goals, and over five
    matches a side's finishing and goalkeeping are part of what it is, not
    noise to be regressed away entirely. `weight` caps how far it moves.
    """
    out = strength.copy()
    done = fixtures[(fixtures["event"] <= as_of_gw) & fixtures["finished"].astype(bool)]
    if done.empty or weight <= 0:
        out["form_games"] = 0
        return out

    rows = []
    for _, f in done.iterrows():
        rows.append({"team": int(f["team_h"]), "gf": f["team_h_score"], "ga": f["team_a_score"]})
        rows.append({"team": int(f["team_a"]), "gf": f["team_a_score"], "ga": f["team_h_score"]})
    r = pd.DataFrame(rows).dropna()
    agg = r.groupby("team").agg(gf=("gf", "mean"), ga=("ga", "mean"), n=("gf", "size"))

    lg_gf = agg["gf"].mean()
    lg_ga = agg["ga"].mean()
    att_form = out.index.map(agg["gf"] / lg_gf)
    def_form = out.index.map(agg["ga"] / lg_ga)
    n = pd.Series(out.index.map(agg["n"]), index=out.index).fillna(0.0)

    # Shrink by sample size, then cap by `weight`.
    w = (n / (n + C.FORM_PRIOR_GAMES)).clip(0, 1) * weight
    for side in ("home", "away"):
        out[f"att_{side}"] = (
            (1 - w) * out[f"att_{side}"] + w * pd.Series(att_form, index=out.index).fillna(1.0)
        )
        out[f"def_{side}"] = (
            (1 - w) * out[f"def_{side}"] + w * pd.Series(def_form, index=out.index).fillna(1.0)
        )
    out["form_games"] = n
    return out


# -------------------------------------------------------------- ensemble
def ensemble_score(
    ep: pd.DataFrame, hist: pd.DataFrame, as_of_gw: int,
    w_own: float = C.ENS_W_OWN, w_ppg: float = C.ENS_W_PPG,
    col: str = "true_total",
) -> pd.Series:
    """Blend the structural projection with two cheap market signals.

    Points-per-game so far and current ownership each predict next week
    independently of the engine -- they carry information the engine has
    no column for: set-piece duty changing, a manager's trust, a role
    switch, a penalty taker swap. Leave-one-gameweek-out validation across
    GW2-5 puts the blend at r = 0.401 against 0.377 for the engine alone,
    improving in every held-out week.

    Returned on the original points scale (mean and spread matched), so
    captaincy, bench weights and the 4-point hit test still mean something.
    """
    prev = hist[hist["round"] <= as_of_gw]
    ppg = prev.groupby("element")["total_points"].mean()
    sel = prev[prev["round"] == as_of_gw].set_index("element")["selected"]

    d = ep.set_index("id")
    base = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    def z(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    own_s = pd.Series(d.index.map(sel), index=d.index)
    ppg_s = pd.Series(d.index.map(ppg), index=d.index)
    mix = z(base) + w_own * z(own_s) + w_ppg * z(ppg_s)
    # Rescale onto the points scale of `col`.
    out = mix * (base.std() / mix.std() if mix.std() > 0 else 1.0) + base.mean()
    return out.clip(lower=0.0).reindex(ep["id"]).to_numpy()
