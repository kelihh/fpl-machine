"""Ingestion.

Two sources, both FPL API shaped:
  live  -- fantasy.premierleague.com bootstrap-static + fixtures
  prior -- last season's end-of-season snapshot from the vaastav archive

They join on `code`, which is a stable player identifier across seasons
(unlike `id`, which is reassigned every year).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

from . import config as C


# ------------------------------------------------------------------ live
def fetch_live(max_age_hours: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (players, teams, fixtures, meta) from the live FPL API.

    Cached to disk so repeated runs in one session don't hammer the API.
    """
    bs = _cached_json(C.CACHE / "bootstrap.json", C.FPL_BOOTSTRAP, max_age_hours)
    fx = _cached_json(C.CACHE / "fixtures.json", C.FPL_FIXTURES, max_age_hours)

    players = pd.DataFrame(bs["elements"])
    teams = pd.DataFrame(bs["teams"])
    fixtures = pd.DataFrame(fx)

    events = pd.DataFrame(bs["events"])
    upcoming = events[~events["finished"].astype(bool)]
    next_gw = int(upcoming["id"].min()) if len(upcoming) else int(events["id"].max())
    meta = {"next_gw": next_gw, "pulled": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

    return players, teams, fixtures, meta


def _cached_json(path: Path, url: str, max_age_hours: float):
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(path.read_text())
    resp = requests.get(url, timeout=30, headers={"User-Agent": "fpl-model/1.0"})
    resp.raise_for_status()
    path.write_text(json.dumps(resp.json()))
    return resp.json()


# ----------------------------------------------------------------- prior
def fetch_prior(repo_dir: Path | None = None) -> pd.DataFrame:
    """Last season's end-of-season player snapshot.

    Sparse-clones the archive if it isn't already on disk.
    """
    repo_dir = repo_dir or (C.CACHE / "archive")
    csv_path = repo_dir / "data" / C.PRIOR_SEASON / "players_raw.csv"

    if not csv_path.exists():
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", C.PRIOR_REPO, str(repo_dir)],
                check=True, capture_output=True,
            )
        subprocess.run(
            ["git", "sparse-checkout", "set", f"data/{C.PRIOR_SEASON}"],
            cwd=repo_dir, check=True, capture_output=True,
        )

    return pd.read_csv(csv_path)


# ------------------------------------------------------------------ join
PRIOR_COLS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "saves", "bonus", "bps", "total_points",
    "expected_goals", "expected_assists", "expected_goals_conceded",
    "defensive_contribution", "goals_conceded",
]


def build_player_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Live players enriched with prior-season per-90 rates."""
    live, teams, fixtures, meta = fetch_live()
    prior = fetch_prior()

    keep = ["code"] + [c for c in PRIOR_COLS if c in prior.columns]
    prior = prior[keep].copy()
    prior.columns = ["code"] + [f"prior_{c}" for c in keep[1:]]

    df = live.merge(prior, on="code", how="left")

    # Per-90 rates from prior season. Guard against tiny denominators --
    # a player with 40 minutes should not define a rate.
    mins = df["prior_minutes"].fillna(0.0)
    safe = mins.where(mins >= 270).astype("float64")
    for src, dst in [
        ("prior_expected_goals", "p_xg90"),
        ("prior_expected_assists", "p_xa90"),
        ("prior_expected_goals_conceded", "p_xgc90"),
        ("prior_defensive_contribution", "p_dc90"),
        ("prior_saves", "p_saves90"),
        ("prior_bonus", "p_bonus90"),
        ("prior_goals_conceded", "p_gc90"),
    ]:
        df[dst] = pd.to_numeric(df[src], errors="coerce") / safe * 90.0
        df[dst] = pd.to_numeric(df[dst], errors="coerce")

    df["prior_mins_per_start"] = pd.to_numeric(
        df["prior_minutes"] / df["prior_starts"].replace(0, pd.NA), errors="coerce"
    )
    df["prior_start_rate"] = (df["prior_starts"].fillna(0) / 38.0).clip(0, 1)

    df["pos"] = df["element_type"].map(C.POS)
    df["price"] = df["now_cost"]
    return df, teams, fixtures, meta
