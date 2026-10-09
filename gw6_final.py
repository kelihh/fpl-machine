"""Detail the shortlisted GW6 squads: XI, bench order, fixture concentration."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fplmodel.data import build_player_frame
from fplmodel.season import blend_rates, load_history
from fplmodel.teams import team_strength, fixture_expectations
from fplmodel.engine import expected_points
from fplmodel.simulate import simulate_xi, summarise

GW = 6
base, teams, fixtures, meta = build_player_frame()
hist = load_history()
df = blend_rates(base, hist, GW - 1)
st = team_strength(df, teams)
fx = fixture_expectations(st, fixtures, GW)
detail = expected_points(df, fx, st, detail=True)
ep = pd.read_csv("cache/gw6_ep.csv").set_index("id")

CUR = [1, 497, 445, 87, 388, 204, 4, 453, 94, 13, 481, 68, 411, 346, 106]
GROSS, MBEUMO, TARK = 124, 427, 229
RICE, ANDERSON, THIAW, BARNES = 13, 481, 445, 453
SHAPES = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]

CAND = {
    "A  Rice->Gross only": [i for i in CUR if i != RICE] + [GROSS],
    "B  Rice->Gross + Anderson->Mbeumo": [i for i in CUR if i not in (RICE, ANDERSON)] + [GROSS, MBEUMO],
    "C  Rice->Gross + Thiaw->Tarkowski": [i for i in CUR if i not in (RICE, THIAW)] + [GROSS, TARK],
    "F  Rice->Gross + Barnes->Mbeumo": [i for i in CUR if i not in (RICE, BARNES)] + [GROSS, MBEUMO],
}


def ok(i):
    s, c = ep.at[i, "status"], ep.at[i, "chance_of_playing_next_round"]
    return s == "a" and (pd.isna(c) or c >= 100)


def best_xi(ids, max_fix=None):
    pool = [i for i in ids if ok(i)]
    byp = {p: sorted([i for i in pool if ep.at[i, "pos"] == p],
                     key=lambda i: -ep.at[i, "ens"]) for p in ("GK", "DEF", "MID", "FWD")}
    best = None
    for nd, nm, nf in SHAPES:
        need = {"GK": 1, "DEF": nd, "MID": nm, "FWD": nf}
        if any(len(byp[p]) < n for p, n in need.items()):
            continue
        xi = [i for p, n in need.items() for i in byp[p][:n]]
        if max_fix:
            c = {}
            for i in xi:
                c[ep.at[i, "fixid"]] = c.get(ep.at[i, "fixid"], 0) + 1
            if max(c.values()) > max_fix:
                continue
        tot = sum(ep.at[i, "ens"] for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{nd}-{nm}-{nf}")
    return best


for label, squad in CAND.items():
    b = best_xi(squad)
    xi = b[1]
    cap = max(xi, key=lambda i: ep.at[i, "ens"])
    vice = sorted(xi, key=lambda i: -ep.at[i, "ens"])[1]
    bench = [i for i in squad if i not in xi]
    bench = sorted(bench, key=lambda i: (ep.at[i, "pos"] == "GK", -ep.at[i, "ens"]))
    cost = ep.loc[squad, "price"].sum() / 10
    print(f"\n{'='*78}\n{label}   shape {b[2]}   squad £{cost}m   bank £{round(99.9-cost,1)}m")
    t = ep.loc[xi, ["web_name", "club", "pos", "price", "opp", "ha", "p_start", "ens", "selected_by_percent"]].copy()
    t["role"] = ["(C)" if i == cap else "(V)" if i == vice else "" for i in xi]
    t["fixture"] = [ep.at[i, "fixid"] for i in xi]
    print(t.round(2).to_string())
    print("BENCH:", " > ".join(f"{ep.at[i,'web_name']} ({ep.at[i,'pos']} {ep.at[i,'ens']:.1f})" for i in bench))
    tc = pd.Series([ep.at[i, "club"] for i in xi]).value_counts()
    print("same-side stacks in XI:", dict(tc[tc > 1]), " max per club =", int(tc.max()))
    tot, pp = simulate_xi(detail, xi, cap, n_sims=60_000, seed=11)
    print(summarise(tot, label))
    tot3, _ = simulate_xi(detail, xi, cap, n_sims=60_000, seed=11, triple=True)
    print("triple captain on", ep.at[cap, "web_name"], ":", summarise(tot3, "TC"))
