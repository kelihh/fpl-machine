"""Compare candidate GW6 squads on the distribution, not the mean."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fplmodel.data import build_player_frame
from fplmodel.season import blend_rates, load_history, ensemble_score
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
ep = pd.read_csv("cache/gw6_ep.csv")

NM = dict(zip(ep["id"], ep["web_name"]))
FI = dict(zip(ep["id"], ep["fixid"]))
S = dict(zip(ep["id"], ep["ens"]))
P = dict(zip(ep["id"], ep["pos"]))
OPP = dict(zip(ep["id"], ep["opp"] + "(" + ep["ha"] + ")"))

CUR = [1, 497, 445, 87, 388, 204, 4, 453, 94, 13, 481, 68, 411, 346, 106]
GROSS, MBEUMO, TARK, ROGERS, DECUYPER, HALL = 124, 427, 229, 40, 115, 449
RICE, ANDERSON, THIAW, THIAGO, BARNES = 13, 481, 445, 106, 453

CAND = {
    "A  Rice->Gross (roll 1 FT)": [i for i in CUR if i != RICE] + [GROSS],
    "B  Rice->Gross, Anderson->Mbeumo": [i for i in CUR if i not in (RICE, ANDERSON)] + [GROSS, MBEUMO],
    "C  Rice->Gross, Thiaw->Tarkowski": [i for i in CUR if i not in (RICE, THIAW)] + [GROSS, TARK],
    "D  Rice->Gross, Thiago->Tarkowski*": [i for i in CUR if i not in (RICE, THIAGO)] + [GROSS, TARK],
    "E  Rice->Mbeumo, Thiaw->Gross": [i for i in CUR if i not in (RICE, THIAW)] + [MBEUMO, GROSS],
    "F  Rice->Gross, Barnes->Mbeumo": [i for i in CUR if i not in (RICE, BARNES)] + [GROSS, MBEUMO],
}

SHAPES = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]
flag = dict(zip(ep["id"], zip(ep["status"], ep["chance_of_playing_next_round"])))


def ok(i):
    s, c = flag[i]
    return s == "a" and (pd.isna(c) or c >= 100)


def best_xi(ids, max_fix=None):
    pool = [i for i in ids if ok(i)]
    byp = {p: sorted([i for i in pool if P[i] == p], key=lambda i: -S[i])
           for p in ("GK", "DEF", "MID", "FWD")}
    best = None
    for nd, nm, nf in SHAPES:
        need = {"GK": 1, "DEF": nd, "MID": nm, "FWD": nf}
        if any(len(byp[p]) < n for p, n in need.items()):
            continue
        xi = [i for p, n in need.items() for i in byp[p][:n]]
        if max_fix:
            c = {}
            for i in xi:
                c[FI[i]] = c.get(FI[i], 0) + 1
            if max(c.values()) > max_fix:
                continue
        tot = sum(S[i] for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{nd}-{nm}-{nf}")
    return best


rows = []
for label, squad in CAND.items():
    price = ep.set_index("id").loc[squad, "price"].sum()
    for cap_fix in (None, 2):
        b = best_xi(squad, max_fix=cap_fix)
        if b is None:
            continue
        xi = b[1]
        cap = max(xi, key=lambda i: S[i])
        tot, _ = simulate_xi(detail, xi, cap, n_sims=40_000, seed=11)
        s = summarise(tot, label)
        s["cap_fix"] = cap_fix or "-"
        s["shape"] = b[2]
        s["£"] = price / 10
        s["captain"] = NM[cap]
        rows.append(s)

r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(r[["squad", "cap_fix", "shape", "£", "captain", "mean", "sd", "p10", "median", "p90", "P<40", "P>70"]].to_string(index=False))
