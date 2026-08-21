"""Squad optimisation.

Binary integer program over the full player pool. Two linked decisions:
squad membership (15) and starting XI (11), plus a captain. Solved exactly
by CBC in well under a second -- no shortlisting, which matters because
pre-filtering to a human-picked candidate list quietly reinjects the bias
the model exists to remove.
"""
from __future__ import annotations

import pandas as pd
import pulp

from . import config as C


def optimise_squad(
    ep: pd.DataFrame,
    budget: int = C.BUDGET,
    score_col: str = "horizon",
    locked: list[int] | None = None,
    banned: list[int] | None = None,
) -> dict:
    """Pick the optimal 15, XI, captain and bench order."""
    df = ep[ep["status"] != "u"].reset_index(drop=True)
    ids = df["id"].tolist()
    score = dict(zip(ids, df[score_col]))
    price = dict(zip(ids, df["price"]))
    pos = dict(zip(ids, df["pos"]))
    club = dict(zip(ids, df["team"]))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    sq = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    xi = pulp.LpVariable.dicts("xi", ids, cat="Binary")
    cap = pulp.LpVariable.dicts("cap", ids, cat="Binary")

    # Bench points are worth something but not much.
    bench_w = sum(C.BENCH_WEIGHT) / len(C.BENCH_WEIGHT)
    prob += pulp.lpSum(
        score[i] * xi[i] + score[i] * cap[i] + score[i] * bench_w * (sq[i] - xi[i])
        for i in ids
    )

    prob += pulp.lpSum(sq[i] for i in ids) == C.SQUAD_SIZE
    prob += pulp.lpSum(xi[i] for i in ids) == C.XI_SIZE
    prob += pulp.lpSum(cap[i] for i in ids) == 1
    prob += pulp.lpSum(price[i] * sq[i] for i in ids) <= budget

    for i in ids:
        prob += xi[i] <= sq[i]
        prob += cap[i] <= xi[i]

    for p, n in C.POS_QUOTA.items():
        prob += pulp.lpSum(sq[i] for i in ids if pos[i] == p) == n
    for p in C.XI_MIN:
        sel = [xi[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(sel) >= C.XI_MIN[p]
        prob += pulp.lpSum(sel) <= C.XI_MAX[p]

    for c in set(club.values()):
        prob += pulp.lpSum(sq[i] for i in ids if club[i] == c) <= C.MAX_PER_CLUB

    for i in locked or []:
        if i in sq:
            prob += sq[i] == 1
    for i in banned or []:
        if i in sq:
            prob += sq[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver failed: {pulp.LpStatus[prob.status]}")

    chosen = [i for i in ids if sq[i].value() > 0.5]
    starters = [i for i in ids if xi[i].value() > 0.5]
    captain = next(i for i in ids if cap[i].value() > 0.5)

    squad = df[df["id"].isin(chosen)].copy()
    squad["starting"] = squad["id"].isin(starters)
    squad["captain"] = squad["id"] == captain
    squad = squad.sort_values(
        ["starting", "pos", score_col], ascending=[False, True, False]
    )

    return {
        "squad": squad,
        "captain": captain,
        "cost": sum(price[i] for i in chosen),
        # Reported value: what the XI plus captain is projected to score.
        "expected": sum(score[i] for i in starters) + score[captain],
        # Objective value: includes the bench term the solver actually
        # maximised. Sensitivity must compare on this, not on `expected`,
        # or forcing a player in can appear to *improve* the solution.
        "objective": pulp.value(prob.objective),
    }


def suggest_transfers(
    ep: pd.DataFrame, current_ids: list[int], bank: int = 0,
    free_transfers: int = 1, score_col: str = "horizon", hit: float = 4.0,
) -> pd.DataFrame:
    """Rank single swaps by net gain after any points hit.

    A move only earns its hit if the horizon gain exceeds 4 points -- the
    same test a good manager applies by eye, done exhaustively.
    """
    df = ep.set_index("id")
    current = df.loc[[i for i in current_ids if i in df.index]]
    sale_value = current["price"].sum() + bank

    rows = []
    for out_id, out_row in current.iterrows():
        budget = sale_value - (current["price"].sum() - out_row["price"])
        club_counts = current.drop(out_id)["team"].value_counts()

        pool = df[
            (df["pos"] == out_row["pos"])
            & (df["price"] <= budget)
            & (~df.index.isin(current_ids))
            & (df["status"] != "u")
        ]
        pool = pool[pool["team"].map(lambda t: club_counts.get(t, 0)) < C.MAX_PER_CLUB]
        if pool.empty:
            continue

        best = pool[score_col].idxmax()
        gain = pool.at[best, score_col] - out_row[score_col]
        rows.append({
            "out": out_row["web_name"], "out_id": out_id,
            "out_score": round(out_row[score_col], 2),
            "in": pool.at[best, "web_name"], "in_id": best,
            "in_score": round(pool.at[best, score_col], 2),
            "price_delta": (pool.at[best, "price"] - out_row["price"]) / 10,
            "gain": round(gain, 2),
        })

    out = pd.DataFrame(rows).sort_values("gain", ascending=False)
    if out.empty:
        return out
    out["net_after_hit"] = out["gain"] - hit
    out["worth_free_transfer"] = out["gain"] > 0.5
    out["worth_hit"] = out["net_after_hit"] > 0
    return out.reset_index(drop=True)


def sensitivity(ep: pd.DataFrame, result: dict, score_col: str = "horizon",
                n: int = 10, candidates: int = 30) -> pd.DataFrame:
    """True reduced cost for the best players the solver left out.

    For each candidate, re-solve the whole problem with that player forced
    into the squad and measure how much total expected value is lost. This
    is the real cost of overriding the model -- unlike a raw score gap, it
    accounts for the budget and club slots the forced pick consumes.

    A cost near zero means the model is indifferent. That is the point at
    which your own read of the team news is worth more than the output.
    """
    chosen = set(result["squad"]["id"])
    baseline = result["objective"]

    pool = ep[~ep["id"].isin(chosen) & (ep["status"] != "u")]
    pool = pool.nlargest(candidates, score_col)

    rows = []
    for _, r in pool.iterrows():
        try:
            forced = optimise_squad(ep, score_col=score_col, locked=[int(r["id"])])
        except RuntimeError:
            continue
        rows.append({
            "web_name": r["web_name"], "pos": r["pos"],
            "price": r["price"] / 10, score_col: round(r[score_col], 1),
            "reduced_cost": round(baseline - forced["objective"], 2),
            "selected_by_percent": r["selected_by_percent"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.nsmallest(n, "reduced_cost").reset_index(drop=True)
