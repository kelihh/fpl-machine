"""GW6 decision run."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from fplmodel.data import build_player_frame
from fplmodel.season import blend_rates, load_history, ensemble_score
from fplmodel.teams import team_strength, fixture_expectations
from fplmodel.engine import expected_points, horizon_points
import fplmodel.config as C

GW = 6
SQUAD = [1, 497, 445, 87, 388, 204, 4, 453, 94, 13, 481, 68, 411, 346, 106]

base, teams, fixtures, meta = build_player_frame()
hist = load_history()
df = blend_rates(base, hist, GW - 1)          # real flags kept
st = team_strength(df, teams)
tn = dict(zip(teams["id"], teams["short_name"]))

fx = fixture_expectations(st, fixtures, GW)
ep = expected_points(df, fx, st)
hz = horizon_points(df, fixtures, st, GW, 3)
ep = ep.merge(hz[["id", "horizon"]], on="id", how="left")

ep = ep.merge(df[["id","chance_of_playing_next_round"]], on="id", how="left")
ep["ens"] = ensemble_score(ep, hist, GW - 1, col="true_total")
ratio = (ep["ens"] / ep["true_total"].replace(0, np.nan)).fillna(1.0).clip(0.3, 3.0)
ep["hz_ens"] = ep["horizon"] * ratio

# fixture labels
fxg = fixtures[fixtures["event"] == GW]
opp, home = {}, {}
for _, f in fxg.iterrows():
    opp[int(f["team_h"])] = tn[int(f["team_a"])]; home[int(f["team_h"])] = True
    opp[int(f["team_a"])] = tn[int(f["team_h"])]; home[int(f["team_a"])] = False
ep["club"] = ep["team"].map(tn)
ep["opp"] = ep["team"].map(opp)
ep["ha"] = ep["team"].map(home).map({True: "H", False: "A"})
ep["fixid"] = ep["team"].map({int(f[k]): int(f["id"]) for _, f in fxg.iterrows() for k in ("team_h", "team_a")})
ep["xgf"] = ep["team"].map(dict(zip(fx["team"], fx["xg_for"])))
ep["xga"] = ep["team"].map(dict(zip(fx["team"], fx["xg_against"])))
ep.to_csv("cache/gw6_ep.csv", index=False)

pd.set_option("display.width", 260)
cols = ["id", "web_name", "club", "pos", "price", "opp", "ha", "p_start", "true_total",
        "ens", "hz_ens", "status", "selected_by_percent"]
mine = ep[ep["id"].isin(SQUAD)].sort_values(["pos", "ens"], ascending=[True, False])
print("=== CURRENT 15, GW6 ===")
print(mine[cols].round(2).to_string(index=False))
print("\nsquad value", mine["price"].sum() / 10)

print("\n=== TOP 30 BY GW6 ENSEMBLE EP ===")
print(ep.nlargest(30, "ens")[cols].round(2).to_string(index=False))
