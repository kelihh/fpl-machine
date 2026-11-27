"""Rank, not points.

The solver maximises expected points against an imaginary average manager.
In a four-person mini-league that is the wrong objective: what decides the
table is the margin against three specific squads, and those squads share
most of your players. A captain who adds a point of expectation while
every rival owns him adds nothing to your rank.

So simulate the gameweek once and score every entry against the same
drawn scorelines. Shared players cancel, differentials do not, and the
output is P(I finish above each rival) instead of a mean nobody can bank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

API = "https://fantasy.premierleague.com/api"


def _get(path: str) -> dict:
    r = requests.get(f"{API}/{path}", timeout=25,
                     headers={"User-Agent": "fpl-model/1.0"})
    r.raise_for_status()
    return r.json()


def entry_state(entry_id: int) -> dict:
    """Squad, bank, chips and free transfers, read rather than assumed."""
    e = _get(f"entry/{entry_id}/")
    h = _get(f"entry/{entry_id}/history/")
    cur = pd.DataFrame(h["current"])
    used = {c["name"]: c["event"] for c in h.get("chips", [])}
    return {
        "name": e.get("name"),
        "overall_rank": e.get("summary_overall_rank"),
        "total": e.get("summary_overall_points"),
        "bank": e.get("last_deadline_bank", 0),
        "value": e.get("last_deadline_value", 0),
        "chips_used": used,
        "history": cur,
        "bench_wasted": int(cur["points_on_bench"].sum()) if len(cur) else 0,
    }


def picks(entry_id: int, gw: int) -> dict:
    """One entry's fielded team for a completed gameweek."""
    p = _get(f"entry/{entry_id}/event/{gw}/picks/")
    rows = sorted(p["picks"], key=lambda x: x["position"])
    return {
        "xi": [x["element"] for x in rows if x["position"] <= 11],
        "bench": [x["element"] for x in rows if x["position"] > 11],
        "captain": next((x["element"] for x in rows if x["is_captain"]), None),
        "vice": next((x["element"] for x in rows if x["is_vice_captain"]), None),
        "chip": p.get("active_chip"),
    }


def standings(league_id: int) -> pd.DataFrame:
    s = _get(f"leagues-classic/{league_id}/standings/")
    return pd.DataFrame([
        {"rank": r["rank"], "entry": r["entry"], "team": r["entry_name"],
         "manager": r["player_name"], "total": r["total"]}
        for r in s["standings"]["results"]
    ])


# ------------------------------------------------------------------ EO
def effective_ownership(my15: list[int], rivals: dict[int, dict]) -> pd.DataFrame:
    """Ownership and captaincy among the managers who actually decide the table.

    Global ownership is the wrong denominator. A 76%-owned striker whom all
    three rivals captain is 200% effective ownership against you; a 4%-owned
    defender nobody else holds is a genuine differential even though the
    model calls both of them ordinary.
    """
    n = len(rivals)
    rows = {}
    for state in rivals.values():
        for pid in state["xi"]:
            r = rows.setdefault(pid, {"owned": 0, "captained": 0})
            r["owned"] += 1
            if pid == state["captain"]:
                r["captained"] += 1
    out = []
    for pid, r in rows.items():
        out.append({
            "id": pid, "rivals_own": r["owned"], "rivals_captain": r["captained"],
            "rival_eo": round((r["owned"] + r["captained"]) / n * 100, 0),
            "i_own": pid in my15,
        })
    for pid in my15:
        if pid not in rows:
            out.append({"id": pid, "rivals_own": 0, "rivals_captain": 0,
                        "rival_eo": 0.0, "i_own": True})
    return pd.DataFrame(out).sort_values("rival_eo", ascending=False)


# ------------------------------------------------- correlated league sim
def simulate_league(
    detail: pd.DataFrame, me: dict, rivals: dict[int, dict],
    n_sims: int = 60_000, seed: int = 17,
) -> dict:
    """Score every entry against one shared set of simulated scorelines.

    The correlation is the point. Drawing each manager's score independently
    would make shared players a source of variance between entries, when in
    reality they cancel exactly.
    """
    from .simulate import simulate_xi

    entries = {"me": me} | {f"rival_{k}": v for k, v in rivals.items()}
    totals = {}
    for key, state in entries.items():
        xi = [i for i in state["xi"] if i in set(detail["id"])]
        if len(xi) < 8 or state["captain"] not in xi:
            totals[key] = None
            continue
        t, _ = simulate_xi(detail, xi, state["captain"], n_sims=n_sims, seed=seed)
        totals[key] = t

    mine = totals["me"]
    out = {"mean": round(float(mine.mean()), 1)}
    beats = []
    for key, t in totals.items():
        if key == "me" or t is None:
            continue
        rid = int(key.split("_")[1])
        gap = rivals[rid].get("points_gap", 0)   # their season total minus mine
        out[f"p_beat_{rid}"] = round(float((mine - t > gap).mean()) * 100, 1)
        beats.append(float((mine - t > gap).mean()))
    out["expected_places_gained"] = round(float(np.sum(beats)), 2)
    out["p_beat_all"] = round(
        float(np.mean([all(mine[i] - t[i] > rivals[int(k.split('_')[1])].get("points_gap", 0)
                           for k, t in totals.items() if k != "me" and t is not None)
                       for i in range(0, n_sims, max(1, n_sims // 20_000))])) * 100, 1)
    return out
