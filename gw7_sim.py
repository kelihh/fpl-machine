"""GW7: candidate squads on the correlated simulation, with and without the
same-club cap I imposed after GW5. GW6 showed the cap never bound; this week
it would bind hard, because the best fixture in the gameweek has three of my
players on one side of it."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fplmodel.data import build_player_frame
from fplmodel.season import blend_rates, load_history
from fplmodel.teams import team_strength, fixture_expectations
from fplmodel.engine import expected_points
from fplmodel.simulate import simulate_xi, summarise

GW = 7
base, teams, fixtures, meta = build_player_frame()
hist = load_history()
df = blend_rates(base, hist, GW - 1)
st = team_strength(df, teams)
fx = fixture_expectations(st, fixtures, GW)
detail = expected_points(df, fx, st, detail=True)
ep = pd.read_csv("cache/gw7_ep.csv").set_index("id")

CUR = [1, 497, 445, 87, 388, 204, 4, 453, 94, 124, 481, 68, 411, 346, 106]
JP, CHERKI, DECUY, CALA, ROGERS, MBEUMO, TARK = 165, 399, 115, 8, 40, 427, 229
THIAGO, ANDERSON, THIAW, AJER, MITCH, CALLEW, SCHADE = 106, 481, 445, 87, 204, 346, 94
BANK = 18

CAND = {
    "A  no transfer":                          CUR,
    "B  Thiago>JoaoPedro":                     [i for i in CUR if i != THIAGO] + [JP],
    "C  Thiago>JP + Anderson>Cherki":          [i for i in CUR if i not in (THIAGO, ANDERSON)] + [JP, CHERKI],
    "D  Thiago>JP + Thiaw>De Cuyper":          [i for i in CUR if i not in (THIAGO, THIAW)] + [JP, DECUY],
    "E  Thiago>JP + Thiaw>Calafiori":          [i for i in CUR if i not in (THIAGO, THIAW)] + [JP, CALA],
    "F  Anderson>Cherki + Thiaw>De Cuyper":    [i for i in CUR if i not in (ANDERSON, THIAW)] + [CHERKI, DECUY],
    "G  Thiago>JP + CalLew>De Cuyper*":        [i for i in CUR if i not in (THIAGO, CALLEW)] + [JP, DECUY],
}

SHAPES = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]


def ok(i):
    s, c = ep.at[i, "status"], ep.at[i, "chance_of_playing_next_round"]
    return s == "a" and (pd.isna(c) or c >= 100)


def best_xi(ids, cap_club=None):
    pool = [i for i in ids if ok(i)]
    byp = {p: sorted([i for i in pool if ep.at[i, "pos"] == p],
                     key=lambda i: -ep.at[i, "ens"]) for p in ("GK", "DEF", "MID", "FWD")}
    best = None
    for nd, nm, nf in SHAPES:
        need = {"GK": 1, "DEF": nd, "MID": nm, "FWD": nf}
        if any(len(byp[p]) < n for p, n in need.items()):
            continue
        xi = [i for p, n in need.items() for i in byp[p][:n]]
        if cap_club:
            cc = {}
            for i in xi:
                cc[ep.at[i, "club"]] = cc.get(ep.at[i, "club"], 0) + 1
            if max(cc.values()) > cap_club:
                continue
        tot = sum(ep.at[i, "ens"] for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{nd}-{nm}-{nf}")
    return best


def legal(ids):
    pc = pd.Series([ep.at[i, "pos"] for i in ids]).value_counts().to_dict()
    if pc != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}:
        bad = {k: pc.get(k, 0) for k in ("GK", "DEF", "MID", "FWD")}
        return f"pos {bad}"
    cl = pd.Series([ep.at[i, "team"] for i in ids]).value_counts()
    if cl.max() > 3:
        return "club>3"
    return None


rows = []
for lab, sq in CAND.items():
    cost = ep.loc[sq, "price"].sum()
    issue = legal(sq)
    for cap in (None, 2):
        b = best_xi(sq, cap_club=cap)
        if b is None:
            continue
        xi = b[1]
        c = max(xi, key=lambda i: ep.at[i, "ens"])
        t, _ = simulate_xi(detail, xi, c, n_sims=60_000, seed=77)
        s = summarise(t, lab)
        s["cap"] = cap or "none"
        s["shape"] = b[2]
        s["£"] = cost / 10
        s["ok"] = "OVER" if cost > 99.9 else (issue or "ok")
        rows.append(s)
    # triple captain, unconstrained
    b = best_xi(sq)
    xi = b[1]
    c = max(xi, key=lambda i: ep.at[i, "ens"])
    t3, _ = simulate_xi(detail, xi, c, n_sims=60_000, seed=77, triple=True)
    s = summarise(t3, lab)
    s.update({"cap": "none+TC", "shape": b[2], "£": cost / 10,
              "ok": "OVER" if cost > 99.9 else (issue or "ok")})
    rows.append(s)

r = pd.DataFrame(rows)
pd.set_option("display.width", 230)
print(r[["squad", "ok", "£", "cap", "shape", "mean", "sd", "p10", "median", "p90", "P<40", "P>70"]].to_string(index=False))
