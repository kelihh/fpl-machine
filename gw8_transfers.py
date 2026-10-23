"""Exhaustive 1- and 2-transfer search for GW8."""
import warnings; warnings.filterwarnings("ignore")
import itertools, numpy as np, pandas as pd

ep = pd.read_csv("cache/gw8_ep.csv")
SQUAD = [1, 497, 115, 87, 388, 204, 4, 453, 94, 124, 481, 68, 411, 346, 106]
BANK = 18
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
POSQ = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

P = dict(zip(ep["id"], ep["pos"]))
PR = dict(zip(ep["id"], ep["price"]))
TM = dict(zip(ep["id"], ep["team"]))
FI = dict(zip(ep["id"], ep["fixid"]))
NM = dict(zip(ep["id"], ep["web_name"]))
chance = dict(zip(ep["id"], ep["chance_of_playing_next_round"]))
stat = dict(zip(ep["id"], ep["status"]))
SHAPES = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6)
          if 1 <= 10 - d - m <= 3]


def flagged(i):
    c = chance.get(i)
    return stat.get(i) != "a" or (c is not None and not pd.isna(c) and c < 100)


def make_xi(ids, S, no_flags=True, max_fix=None):
    pool = [i for i in ids if not (no_flags and flagged(i))]
    byp = {}
    for p in ("GK", "DEF", "MID", "FWD"):
        byp[p] = sorted([i for i in pool if P[i] == p], key=lambda i: -S[i])
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
        tot = sum(S[i] for i in xi) + max(S[i] for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{nd}-{nm}-{nf}")
    return best


def legal(ids):
    cp = {}
    for i in ids:
        cp[P[i]] = cp.get(P[i], 0) + 1
        if cp[P[i]] > POSQ[P[i]]:
            return False
    ct = {}
    for i in ids:
        ct[TM[i]] = ct.get(TM[i], 0) + 1
        if ct[TM[i]] > 3:
            return False
    return True


BAN = []   # Joao Pedro cleared: 71 min and 12 pts in GW6
avail = ep[(ep["status"] == "a") & (ep["p_start"] > 0.30)
           & (~ep["id"].isin(SQUAD)) & (~ep["id"].isin(BAN))]
budget0 = sum(PR[i] for i in SQUAD) + BANK

for col in ["ens", "hz_ens"]:
    S = dict(zip(ep["id"], ep[col]))
    cur = make_xi(SQUAD, S)
    print(f"\n######## objective = {col}   no transfer: {cur[0]:.2f}  ({cur[2]})")
    topn = {p: avail[avail["pos"] == p].nlargest(26, col)["id"].tolist()
            for p in ("GK", "DEF", "MID", "FWD")}
    rows = []
    for n in (1, 2):
        for outs in itertools.combinations(SQUAD, n):
            rest = [i for i in SQUAD if i not in outs]
            spend = budget0 - sum(PR[i] for i in rest)
            pools = [topn[P[i]] for i in outs]
            for ins in itertools.product(*pools):
                if len(set(ins)) != n or sum(PR[i] for i in ins) > spend:
                    continue
                new = rest + list(ins)
                if not legal(new):
                    continue
                b = make_xi(new, S)
                if b is None:
                    continue
                rows.append({"n": n, "out": " + ".join(NM[i] for i in outs),
                             "in": " + ".join(NM[i] for i in ins),
                             "cost": round((sum(PR[i] for i in ins) - sum(PR[i] for i in outs)) / 10, 1),
                             "xi": round(b[0], 2), "gain": round(b[0] - cur[0], 2),
                             "shape": b[2], "ids_in": list(ins), "ids_out": list(outs)})
    r = pd.DataFrame(rows)
    for n in (1, 2):
        print(f"\n-- best {n} transfer(s)")
        print(r[r["n"] == n].nlargest(10, "gain")[
            ["out", "in", "cost", "xi", "gain", "shape"]].to_string(index=False))
    r.to_csv(f"cache/gw8_tr_{col}.csv", index=False)

# ---- explicit no-transfer XI + captain from the current 15, both objectives
print("\n\n######## NO-TRANSFER XI FROM CURRENT 15 ########")
for col in ["ens", "hz_ens"]:
    S = dict(zip(ep["id"], ep[col]))
    tot, xi, shape = make_xi(SQUAD, S)
    cap = max(xi, key=lambda i: S[i])
    vice = max([i for i in xi if i != cap], key=lambda i: S[i])
    bench = sorted([i for i in SQUAD if i not in xi and P[i] != "GK"],
                   key=lambda i: -S[i])
    benchgk = [i for i in SQUAD if i not in xi and P[i] == "GK"]
    print(f"\n-- objective {col}   shape {shape}   XI+capt score {tot:.2f}")
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for i in sorted(xi, key=lambda i: (order[P[i]], -S[i])):
        tag = " (C)" if i == cap else (" (V)" if i == vice else "")
        print(f"   {P[i]:4} {NM[i]:16}{tag:5} {S[i]:6.2f}")
    print("   bench order:", ", ".join(f"{NM[i]} {S[i]:.2f}" for i in bench),
          "| reserve GK:", ", ".join(NM[i] for i in benchgk))
