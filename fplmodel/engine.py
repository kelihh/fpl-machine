"""Expected points.

The model never assumes an event happens. It computes the probability of
each scoring event and sums the expectations. Two outputs per player:

  if_start   -- points conditional on starting the match
  true_total -- if_start weighted by the probability they actually play

Deliberately excluded: red cards, own goals, penalty misses. All three are
low-frequency noise that distorts minutes-based estimates more than they
add predictive value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import config as C


# ------------------------------------------------------------- minutes
def minutes_model(df: pd.DataFrame) -> pd.DataFrame:
    """P(start), P(60+ | start), and expected minutes.

    Availability flags from the API override the historical rate --
    a 0% chance-of-playing beats any prior.
    """
    out = df.copy()

    start_rate = out["prior_start_rate"].fillna(0.0)

    # Shrink toward a low positional prior when the sample is thin.
    mins = out["prior_minutes"].fillna(0.0)
    lam = mins / (mins + C.PRIOR_STRENGTH)
    out["p_start"] = lam * start_rate + (1 - lam) * 0.25

    # In-season form takes over once there are minutes on the board.
    if out["starts"].fillna(0).sum() > 0:
        played = out["minutes"].fillna(0)
        gw_played = max(int(out["starts"].fillna(0).max()), 1)
        current = (out["starts"].fillna(0) / gw_played).clip(0, 1)
        conf = (played / (played + 270)).clip(0, 1)
        out["p_start"] = (1 - conf) * out["p_start"] + conf * current

    # Availability override.
    chance = out["chance_of_playing_next_round"]
    out.loc[chance.notna(), "p_start"] *= chance[chance.notna()] / 100.0
    out.loc[out["status"].isin(["i", "s", "u", "n"]), "p_start"] = 0.0

    # P(60+ | start) from average minutes per start. A player averaging
    # 64 min/start is far likelier to be hooked before 60 than one on 88.
    mps = out["prior_mins_per_start"].fillna(70.0).clip(1, 90)
    out["p60_given_start"] = _p60_from_mps(mps)

    # Cameo appearances: 1 point, no clean sheet, negligible returns.
    out["p_cameo"] = ((1 - out["p_start"]) * 0.18).clip(0, 0.5)
    out.loc[out["status"].isin(["i", "s", "u", "n"]), "p_cameo"] = 0.0

    out["exp_mins_if_start"] = mps
    return out


def _p60_from_mps(mps: pd.Series) -> pd.Series:
    """Map mean minutes-per-start to P(reaching 60 minutes).

    Logistic fit; substitutions cluster around 60-75 so the curve is
    steep through the middle of that range.
    """
    return pd.Series(1.0 / (1.0 + np.exp(-(mps - 62.0) / 6.5)), index=mps.index).clip(0.01, 0.995)


# --------------------------------------------------------------- points
def expected_points(
    players: pd.DataFrame, fixt: pd.DataFrame, strength: pd.DataFrame
) -> pd.DataFrame:
    """Expected points for every player for one gameweek."""
    df = minutes_model(players)

    # Attach fixture context. Doubles produce duplicate rows (intended);
    # teams with a blank drop out entirely.
    df = df.merge(fixt, left_on="team", right_on="team", how="inner")

    # League baselines for scaling player rates to this fixture.
    lg_xg = C.LEAGUE_GOALS_PER_TEAM
    att_mult = (df["xg_for"] / lg_xg).clip(0.4, 2.2)

    mins_frac = df["exp_mins_if_start"] / 90.0

    # ---- attacking returns
    # FPL's own expected_goals / expected_assists are Opta-sourced and
    # already aligned to FPL's assist definition, so no Understat-style
    # correction factor is applied here.
    pos_goal_pts = df["pos"].map(C.GOAL_POINTS)
    xg = df["p_xg90"].fillna(0.0) * mins_frac * att_mult
    xa = df["p_xa90"].fillna(0.0) * mins_frac * att_mult
    df["pts_goals"] = xg * pos_goal_pts
    df["pts_assists"] = xa * C.ASSIST_POINTS

    # ---- clean sheets: Poisson on fixture-adjusted goals against
    lam_gc = df["xg_against"].clip(0.15, 4.0)
    p_cs_match = np.exp(-lam_gc)
    cs_pts = df["pos"].map(C.CLEAN_SHEET_POINTS)
    df["pts_cs"] = p_cs_match * cs_pts * df["p60_given_start"]

    # ---- goals conceded: -1 per 2 conceded, GK and DEF only
    # E[floor(GC/2)] over the Poisson, truncated at a sane upper bound.
    k = np.arange(0, 9)
    pmf = np.array([poisson.pmf(kk, lam_gc) for kk in k])  # (9, n)
    exp_penalty = (pmf * np.floor(k / 2)[:, None]).sum(axis=0)
    is_back = df["pos"].isin(["GK", "DEF"]).to_numpy()
    df["pts_conceded"] = -exp_penalty * is_back * df["p60_given_start"]

    # ---- saves: scaled by how much shooting the opponent generates
    save_mult = (df["xg_against"] / lg_xg).clip(0.4, 2.2)
    df["pts_saves"] = (
        df["p_saves90"].fillna(0.0) * mins_frac * save_mult / C.SAVES_PER_POINT
    ) * (df["pos"] == "GK")

    # ---- defensive contributions: Poisson on the per-90 rate
    dc_lambda = (df["p_dc90"].fillna(0.0) * mins_frac).clip(0, 40)
    thresh = df["pos"].map(C.DEFCON_THRESHOLD)
    p_defcon = 1.0 - poisson.cdf(thresh - 1, dc_lambda)
    df["pts_defcon"] = p_defcon * C.DEFCON_POINTS

    # ---- bonus: prior-season realised bonus rate, fixture-scaled
    df["pts_bonus"] = df["p_bonus90"].fillna(0.0) * mins_frac * att_mult.clip(0.7, 1.4)

    # ---- appearance
    df["pts_appearance"] = 1.0 + df["p60_given_start"]

    parts = ["pts_appearance", "pts_goals", "pts_assists", "pts_cs",
             "pts_conceded", "pts_saves", "pts_defcon", "pts_bonus"]
    df["if_start"] = df[parts].sum(axis=1)

    # Cameo: one appearance point plus a small slice of attacking upside.
    cameo_value = 1.0 + 0.22 * (df["pts_goals"] + df["pts_assists"])
    df["true_total"] = df["p_start"] * df["if_start"] + df["p_cameo"] * cameo_value

    # Collapse doubles, drop blanks.
    keyed = df.groupby("id").agg(
        true_total=("true_total", "sum"),
        if_start=("if_start", "sum"),
        p_start=("p_start", "max"),
        n_fixtures=("fixture", "count"),
    )
    base = players.set_index("id")[
        ["web_name", "pos", "team", "price", "selected_by_percent", "status", "news"]
    ]
    return base.join(keyed, how="left").fillna(
        {"true_total": 0.0, "if_start": 0.0, "p_start": 0.0, "n_fixtures": 0}
    ).reset_index()


def horizon_points(
    players: pd.DataFrame, fixtures: pd.DataFrame, strength: pd.DataFrame,
    start_gw: int, n_gw: int, decay: float = 0.88,
) -> pd.DataFrame:
    """Sum expected points across several gameweeks, discounting later ones.

    Later gameweeks are less certain (injuries, rotation, form shifts), so
    they get geometrically less weight in transfer decisions.
    """
    from .teams import fixture_expectations

    total = None
    for i in range(n_gw):
        gw = start_gw + i
        fx = fixture_expectations(strength, fixtures, gw)
        if fx.empty:
            continue
        ep = expected_points(players, fx, strength)[["id", "true_total", "if_start"]]
        ep = ep.rename(columns={
            "true_total": f"gw{gw}_true", "if_start": f"gw{gw}_start"
        })
        total = ep if total is None else total.merge(ep, on="id", how="outer")

    if total is None:
        raise ValueError(f"No fixtures found from GW{start_gw}")

    true_cols = [c for c in total.columns if c.endswith("_true")]
    weights = np.array([decay ** i for i in range(len(true_cols))])
    total["horizon"] = (total[true_cols].fillna(0).to_numpy() * weights).sum(axis=1)
    return total
