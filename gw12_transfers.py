"""GW12: XI-aware 1- and 2-transfer search with an optional differential tilt.

Unlike solver.suggest_transfers, this scores a swap by the change in the
best legal XI (plus captain), so a bench-only upgrade is valued as the
insurance it is rather than as a raw score difference.

  python3 gw12_transfers.py <bank_m> <tilt>     tilt 0 = max points, 0.5 = moderate
"""
import warnings; warnings.filterwarnings("ignore")
import itertools, sys, numpy as np, pandas as pd

ep = pd.read_csv("cache/gw12_ep.csv")
SQUAD = [1, 497, 115, 4, 112, 87, 204, 124, 94, 68, 453, 481, 411, 379, 79]
BANK = int(float(sys.argv[1]) * 10) if len(sys.argv) > 1 else 0
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
POSQ = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SHAPES = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6)
          if 1 <= 10 - d - m <= 3]

own = pd.to_numeric(ep["selected_by_percent"], errors="coerce").fillna(0) / 100
for c in ("ens", "hz_ens"):
    ep[c + "_t"] = ep[c] * (1 - TILT * own)

P = dict(zip(ep["id"], ep["pos"]));  PR = dict(zip(ep["id"], ep["price"]))
TM = dict(zip(ep["id"], ep["team"])); NM = dict(zip(ep["id"], ep["web_name"]))
CL = dict(zip(ep["id"], ep["club"])); OW = dict(zip(ep["id"], ep["selected_by_percent"]))
chance = dict(zip(ep["id"], ep["chance_of_playing_next_round"]))
stat = dict(zip(ep["id"], ep["status"]))

def flagged(i):
    c = chance.get(i)
    return stat.get(i) != "a" or (c is not None and not pd.isna(c) and c < 100)

def make_xi(ids, S, no_flags=True):
    # A flagged player is dropped only if a replacement exists; with one
    # playing GK and a 50% flag there may be no clean alternative.
    pool = [i for i in ids if not (no_flags and flagged(i))]
    byp = {p: sorted([i for i in pool if P[i] == p], key=lambda i: -S[i])
           for p in ("GK", "DEF", "MID", "FWD")}
    if not byp["GK"]:
        byp["GK"] = sorted([i for i in ids if P[i] == "GK"], key=lambda i: -S[i])
    best = None
    for nd, nm, nf in SHAPES:
        need = {"GK": 1, "DEF": nd, "MID": nm, "FWD": nf}
        if any(len(byp[p]) < n for p, n in need.items()):
            continue
        xi = [i for p, n in need.items() for i in byp[p][:n]]
        tot = sum(S[i] for i in xi) + max(S[i] for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{nd}-{nm}-{nf}")
    return best

def legal(ids):
    cp, ct = {}, {}
    for i in ids:
        cp[P[i]] = cp.get(P[i], 0) + 1
        if cp[P[i]] > POSQ[P[i]]: return False
        ct[TM[i]] = ct.get(TM[i], 0) + 1
        if ct[TM[i]] > 3: return False
    return True

print(f"BANK {BANK/10:.1f}m   TILT {TILT}")
for col in ["ens_t", "hz_ens_t"]:
    S = dict(zip(ep["id"], ep[col]))
    cur = make_xi(SQUAD, S)
    print(f"\n######## {col}   NO TRANSFER: {cur[0]:.2f}  ({cur[2]})")
    cap = max(cur[1], key=lambda i: S[i])
    for i in sorted(cur[1], key=lambda i: (P[i], -S[i])):
        print(f"   {'(C)' if i==cap else '   '} {NM[i]:14s} {P[i]:4s} {CL[i]:4s} {S[i]:5.2f}  own {OW[i]}%")
    print("   BENCH:", ", ".join(f"{NM[i]}({'FLAG' if flagged(i) else f'{S[i]:.2f}'})"
          for i in sorted([i for i in SQUAD if i not in cur[1]], key=lambda i: -S[i])))

    avail = ep[(ep["status"] == "a") & (ep["p_start"] > 0.30) & (~ep["id"].isin(SQUAD))]
    budget0 = sum(PR[i] for i in SQUAD) + BANK
    topn = {p: avail[avail["pos"] == p].nlargest(30, col)["id"].tolist()
            for p in ("GK", "DEF", "MID", "FWD")}
    rows = []
    for n in (1, 2):
        for outs in itertools.combinations(SQUAD, n):
            rest = [i for i in SQUAD if i not in outs]
            spend = budget0 - sum(PR[i] for i in rest)
            for ins in itertools.product(*[topn[P[i]] for i in outs]):
                if len(set(ins)) != n or sum(PR[i] for i in ins) > spend: continue
                new = rest + list(ins)
                if not legal(new): continue
                b = make_xi(new, S)
                if b is None: continue
                rows.append({"n": n, "out": " + ".join(NM[i] for i in outs),
                             "in": " + ".join(f"{NM[i]}({CL[i]},{OW[i]}%)" for i in ins),
                             "cost": round((sum(PR[i] for i in ins)-sum(PR[i] for i in outs))/10, 1),
                             "xi": round(b[0], 2), "gain": round(b[0]-cur[0], 2), "shape": b[2]})
    r = pd.DataFrame(rows)
    for n in (1, 2):
        print(f"\n-- best {n} transfer(s)")
        print(r[r["n"] == n].nlargest(8, "gain")[["out","in","cost","xi","gain","shape"]].to_string(index=False))
    r.to_csv(f"cache/gw12_tr_{col}.csv", index=False)
