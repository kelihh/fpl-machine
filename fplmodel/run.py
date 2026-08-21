"""Weekly run.

    python -m fplmodel.run --horizon 5
    python -m fplmodel.run --squad 1,2,3,... --bank 5 --free-transfers 2

Writes CSVs and a markdown brief to output/.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config as C
from .data import build_player_frame
from .engine import horizon_points
from .solver import optimise_squad, sensitivity, suggest_transfers
from .teams import team_strength


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FPL expected-points model")
    ap.add_argument("--horizon", type=int, default=5, help="gameweeks to look ahead")
    ap.add_argument("--gw", type=int, default=None, help="override starting gameweek")
    ap.add_argument("--squad", type=str, default=None, help="comma-separated player ids")
    ap.add_argument("--bank", type=float, default=0.0, help="money in the bank, in millions")
    ap.add_argument("--free-transfers", type=int, default=1)
    ap.add_argument("--lock", type=str, default=None, help="player ids to force in")
    ap.add_argument("--ban", type=str, default=None, help="player ids to exclude")
    args = ap.parse_args(argv)

    players, teams, fixtures, meta = build_player_frame()
    gw = args.gw or meta["next_gw"]
    print(f"GW{gw} | data pulled {meta['pulled']} | {len(players)} players", file=sys.stderr)

    strength = team_strength(players, teams)
    ep = horizon_points(players, fixtures, strength, gw, args.horizon)

    meta_cols = players.set_index("id")[
        ["web_name", "pos", "team", "price", "selected_by_percent", "status", "news"]
    ]
    ep = ep.set_index("id").join(meta_cols).reset_index()
    ep = ep.merge(
        teams[["id", "short_name"]].rename(columns={"id": "team", "short_name": "club"}),
        on="team", how="left",
    )

    ids = lambda s: [int(x) for x in s.split(",")] if s else None
    result = optimise_squad(ep, score_col="horizon",
                            locked=ids(args.lock), banned=ids(args.ban))
    near = sensitivity(ep, result)

    ep.sort_values("horizon", ascending=False).to_csv(
        C.OUTPUT / f"gw{gw}_projections.csv", index=False)
    result["squad"].to_csv(C.OUTPUT / f"gw{gw}_optimal_squad.csv", index=False)

    transfers = None
    if args.squad:
        transfers = suggest_transfers(
            ep, ids(args.squad), bank=int(args.bank * 10),
            free_transfers=args.free_transfers,
        )
        transfers.to_csv(C.OUTPUT / f"gw{gw}_transfers.csv", index=False)

    brief = _write_brief(gw, meta, result, near, transfers, args.horizon)
    (C.OUTPUT / f"gw{gw}_brief.md").write_text(brief)

    # Structured payload for the published page.
    sq = result["squad"]
    payload = {
        "gw": gw,
        "pulled": meta["pulled"],
        "horizon": args.horizon,
        "cost": result["cost"] / 10,
        "projected": round(result["expected"], 1),
        "xi": _rows(sq[sq["starting"]]),
        "bench": _rows(sq[~sq["starting"]]),
        "transfers": (transfers.head(6).to_dict("records")
                      if transfers is not None and not transfers.empty else []),
        "near": near.to_dict("records") if not near.empty else [],
    }
    (C.OUTPUT / f"gw{gw}_data.json").write_text(json.dumps(payload, indent=2, default=str))

    from .publish import render
    render(payload)

    print(brief)
    return 0


def _rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append({
            "name": r["web_name"], "pos": r["pos"], "club": r.get("club", ""),
            "price": r["price"] / 10, "ep": round(r["horizon"], 1),
            "own": r["selected_by_percent"], "captain": bool(r["captain"]),
            "news": (r.get("news") or "")[:80],
        })
    return out


def _write_brief(gw, meta, result, near, transfers, horizon) -> str:
    sq = result["squad"]
    L = [f"# GW{gw} brief", "",
         f"Data pulled {meta['pulled']}. Horizon: GW{gw}-{gw + horizon - 1}.", ""]

    L.append("## Optimal XI")
    L.append("")
    L.append("| Pos | Player | Club | £ | Horizon EP | Own |")
    L.append("|---|---|---|---|---|---|")
    for _, r in sq[sq["starting"]].iterrows():
        name = f"**{r['web_name']} (C)**" if r["captain"] else r["web_name"]
        L.append(f"| {r['pos']} | {name} | {r.get('club','')} | "
                 f"{r['price']/10:.1f} | {r['horizon']:.1f} | {r['selected_by_percent']}% |")

    L += ["", "## Bench", ""]
    for _, r in sq[~sq["starting"]].iterrows():
        L.append(f"- {r['web_name']} ({r['pos']}, £{r['price']/10:.1f}) — {r['horizon']:.1f}")

    L += ["", f"**Cost** £{result['cost']/10:.1f}m · "
              f"**Projected** {result['expected']:.1f} pts over {horizon} GWs", ""]

    if transfers is not None and not transfers.empty:
        L += ["## Transfers", "",
              "| Out | In | £ | Gain | Free? | Worth -4? |",
              "|---|---|---|---|---|---|"]
        for _, r in transfers.head(6).iterrows():
            L.append(f"| {r['out']} | {r['in']} | {r['price_delta']:+.1f} | "
                     f"{r['gain']:+.2f} | {'yes' if r['worth_free_transfer'] else 'no'} | "
                     f"{'yes' if r['worth_hit'] else 'no'} |")
        L.append("")

    L += ["## Sensitivity — near misses", "",
          "A reduced cost near zero means the model is indifferent -- that is "
          "where your own read of the team news is worth more than the output.", "",
          "| Player | Pos | £ | EP | Cost of forcing in | Own |", "|---|---|---|---|---|---|"]
    for _, r in near.iterrows():
        L.append(f"| {r['web_name']} | {r['pos']} | {r['price']:.1f} | "
                 f"{r['horizon']:.1f} | {r['reduced_cost']:.2f} | {r['selected_by_percent']}% |")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
