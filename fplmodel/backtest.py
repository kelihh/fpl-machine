"""Out-of-sample backtest.

For each completed gameweek, rebuild the model using ONLY information
available before that gameweek's deadline, project every player, and
correlate against what actually happened.

Two honesty rules, both easy to violate by accident:

  1. Availability flags (`status`, `chance_of_playing_next_round`) are
     *today's*. Using them to predict GW2 is looking at the answer, so
     they are neutralised here.
  2. Per-90 rates and cumulative minutes are recomputed as of the
     gameweek before, never taken from the live snapshot.

Baselines matter more than the headline number. Ownership and price are
both free and both predict points; a model that cannot beat them is
decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import build_player_frame
from .engine import expected_points
from .season import blend_rates, form_overlay, load_history
from .teams import fixture_expectations, team_strength


def _neutralise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["status"] = "a"
    out["chance_of_playing_next_round"] = np.nan
    return out


def project(
    base: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame,
    hist: pd.DataFrame, gw: int, *, blend: bool = True, form: bool = True,
    k_attack: float = C.BLEND_K_ATTACK, k_defence: float = C.BLEND_K_DEFENCE,
    form_weight: float = C.FORM_WEIGHT, neutralise: bool = True,
    as_of_minutes: bool = True,
) -> pd.DataFrame:
    """Project one gameweek using only pre-deadline information."""
    df = _neutralise(base) if neutralise else base.copy()
    if blend or as_of_minutes:
        df = blend_rates(df, hist, gw - 1, k_attack=k_attack,
                         k_defence=k_defence, rates=blend)
    st = team_strength(df, teams)
    if form:
        st = form_overlay(st, fixtures, gw - 1, weight=form_weight)
    fx = fixture_expectations(st, fixtures, gw)
    return expected_points(df, fx, st)


def score_gw(pred: pd.DataFrame, hist: pd.DataFrame, gw: int) -> dict:
    """Correlations of projection vs actual, plus free baselines."""
    actual = hist[hist["round"] == gw].groupby("element")["total_points"].sum()
    d = pred.set_index("id").copy()
    d["actual"] = d.index.map(actual)
    # Pool: anyone in a squad that played that week and has any season minutes.
    d = d[d["actual"].notna()]
    if len(d) < 30:
        return {"gw": gw, "n": len(d)}
    own = pd.to_numeric(d["selected_by_percent"], errors="coerce").fillna(0)
    return {
        "gw": gw, "n": len(d),
        "r": round(d["true_total"].corr(d["actual"]), 3),
        "rho": round(d["true_total"].corr(d["actual"], method="spearman"), 3),
        "r_own": round(own.corr(d["actual"]), 3),
        "r_price": round(d["price"].corr(d["actual"]), 3),
    }


def by_position(pred: pd.DataFrame, hist: pd.DataFrame, gw: int) -> pd.DataFrame:
    actual = hist[hist["round"] == gw].groupby("element")["total_points"].sum()
    d = pred.set_index("id").copy()
    d["actual"] = d.index.map(actual)
    d = d[d["actual"].notna()]
    return d.groupby("pos").apply(
        lambda g: pd.Series({
            "n": len(g),
            "r": round(g["true_total"].corr(g["actual"]), 3) if len(g) > 5 else np.nan,
        }),
        include_groups=False,
    )


def run(gws: list[int], **kw) -> pd.DataFrame:
    base, teams, fixtures, meta = build_player_frame()
    hist = load_history()
    rows = [score_gw(project(base, teams, fixtures, hist, gw, **kw), hist, gw) for gw in gws]
    return pd.DataFrame(rows)
