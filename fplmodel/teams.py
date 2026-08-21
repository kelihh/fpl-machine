"""Team attack and defence strength.

Two independent estimates, blended:

  xG-derived  -- from last season's aggregated player xG / xGC. Accurate
                 but unavailable for promoted sides.
  FPL ratings -- teams.strength_attack_* / strength_defence_*. Coarse and
                 hand-set by FPL, but defined for all 20 clubs including
                 promoted ones, which is exactly where the xG estimate
                 has nothing to say.

Promoted sides fall back to FPL ratings alone. That is the single largest
known weakness in this model -- see README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def team_strength(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team with attack/defence multipliers around 1.0."""
    out = teams[["id", "short_name", "name"]].copy()

    # ---- xG-derived, from prior-season minutes-weighted aggregates
    p = players.dropna(subset=["prior_minutes"]).copy()
    p = p[p["prior_minutes"] > 0]

    agg = p.groupby("team").apply(
        lambda g: pd.Series({
            "xg_for": g["prior_expected_goals"].sum(),
            "xgc_w": np.average(
                g["p_xgc90"].fillna(0), weights=g["prior_minutes"].clip(lower=1)
            ) if g["p_xgc90"].notna().any() else np.nan,
            "mins": g["prior_minutes"].sum(),
        }),
        include_groups=False,
    )

    # Squad minutes ~ 38 matches x 11 players x 90. Scale xG to per-match.
    agg["xg_per_match"] = agg["xg_for"] / (agg["mins"] / (11 * 90)).clip(lower=1)
    agg["xgc_per_match"] = agg["xgc_w"]

    out = out.merge(
        agg[["xg_per_match", "xgc_per_match"]],
        left_on="id", right_index=True, how="left",
    )

    # Only teams with a meaningful prior sample get an xG estimate.
    prior_mins = p.groupby("team")["prior_minutes"].sum()
    out["has_prior"] = out["id"].map(prior_mins).fillna(0) > 20_000

    lg_att = out.loc[out["has_prior"], "xg_per_match"].mean()
    lg_def = out.loc[out["has_prior"], "xgc_per_match"].mean()
    out["att_xg"] = (out["xg_per_match"] / lg_att).where(out["has_prior"])
    out["def_xg"] = (out["xgc_per_match"] / lg_def).where(out["has_prior"])

    # ---- FPL ordinal ratings, normalised to mean 1.0
    # These are all zero until the season starts, so fall back through
    # progressively coarser fields rather than dividing by zero.
    t = teams.set_index("id")
    for side in ("home", "away"):
        a = _usable(t, [f"strength_attack_{side}", f"strength_overall_{side}", "strength"])
        d = _usable(t, [f"strength_defence_{side}", f"strength_overall_{side}", "strength"])
        out[f"att_fpl_{side}"] = out["id"].map(a / a.mean()) if a is not None else 1.0
        # Higher FPL defence strength = better defence = fewer goals
        # conceded, so invert to keep "higher means leakier".
        out[f"def_fpl_{side}"] = out["id"].map(d.mean() / d) if d is not None else 1.0

    # ---- blend
    w = np.where(out["has_prior"], C.XG_PRIOR_WEIGHT, 0.0)
    for side in ("home", "away"):
        out[f"att_{side}"] = (
            w * out["att_xg"].fillna(1.0) + (1 - w) * out[f"att_fpl_{side}"]
        )
        out[f"def_{side}"] = (
            w * out["def_xg"].fillna(1.0) + (1 - w) * out[f"def_fpl_{side}"]
        )

    return out.set_index("id")


def fixture_expectations(
    strength: pd.DataFrame, fixtures: pd.DataFrame, gw: int
) -> pd.DataFrame:
    """Expected goals for and against, per team, for one gameweek.

    Handles blanks (team absent) and doubles (two rows, summed downstream).
    """
    fx = fixtures[fixtures["event"] == gw]
    rows = []
    for _, f in fx.iterrows():
        h, a = int(f["team_h"]), int(f["team_a"])
        # Attack of one side scaled by leakiness of the other.
        h_xg = C.LEAGUE_GOALS_PER_TEAM * strength.at[h, "att_home"] * strength.at[a, "def_away"]
        a_xg = C.LEAGUE_GOALS_PER_TEAM * strength.at[a, "att_away"] * strength.at[h, "def_home"]
        h_xg *= C.HOME_BOOST
        a_xg *= C.AWAY_PENALTY
        rows.append({"team": h, "opp": a, "is_home": True,
                     "xg_for": h_xg, "xg_against": a_xg, "fixture": f["id"]})
        rows.append({"team": a, "opp": h, "is_home": False,
                     "xg_for": a_xg, "xg_against": h_xg, "fixture": f["id"]})
    return pd.DataFrame(rows)


def _usable(t: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """First strength column that is actually populated and varies.

    FPL zeroes out the granular attack/defence ratings until the season
    is underway, so preseason runs need the coarser `strength` field.
    """
    for col in candidates:
        if col not in t.columns:
            continue
        s = pd.to_numeric(t[col], errors="coerce").astype(float)
        if s.notna().all() and s.mean() > 0 and s.std() > 0:
            return s
    return None
