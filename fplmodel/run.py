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
from .season import blend_rates, ensemble_score, fetch_history
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
    ap.add_argument("--differential", type=float, default=0.0,
                    help="0=maximise points, 0.5=moderate tilt away from template, "
                         "1=aggressive. Useful when you are behind and need variance.")
    args = ap.parse_args(argv)

    players, teams, fixtures, meta = build_player_frame()
    gw = args.gw or meta["next_gw"]
    print(f"GW{gw} | data pulled {meta['pulled']} | {len(players)} players", file=sys.stderr)

    # Current-season evidence. Without this the scheduled run uses only
    # prior-season per-90 rates and publishes a materially different model
    # from the one the backtest measures: 0.370 against 0.430.
    hist = None
    if gw > 1:
        try:
            hist = fetch_history()
            players = blend_rates(players, hist, gw - 1)
            print(f"blended {hist['round'].max()} gameweeks of current-season form",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 -- never fail the weekly run
            print(f"WARNING: current-season blend unavailable ({exc}); "
                  "falling back to prior-season rates only", file=sys.stderr)

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

    # FPL is a rank game. Maximising points maximises expected score against
    # the field, but ownership drives VARIANCE around it: template narrows
    # your spread, differentials widen it. Tilt only when you need variance.
    # Market ensemble: points-per-game and ownership carry information the
    # engine has no column for -- set-piece duty changing, a manager's trust,
    # a penalty-taker swap. Leave-one-gameweek-out validated.
    if hist is not None:
        try:
            ep["horizon"] = pd.Series(
                ensemble_score(ep, hist, gw - 1, col="horizon"), index=ep.index)
            print("applied market ensemble to the horizon", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: ensemble unavailable ({exc})", file=sys.stderr)

    own = pd.to_numeric(ep["selected_by_percent"], errors="coerce").fillna(0) / 100
    ep["obj"] = ep["horizon"] * (1 - args.differential * own)
    col = "obj"

    result = optimise_squad(ep, score_col=col,
                            locked=ids(args.lock), banned=ids(args.ban))
    near = sensitivity(ep, result, score_col=col)

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
        "sim": _simulate(players, teams, fixtures, gw, sq),
    }
    (C.OUTPUT / f"gw{gw}_data.json").write_text(json.dumps(payload, indent=2, default=str))

    from .publish import render
    render(payload)

    print(brief)
    return 0


def _simulate(players, teams, fixtures, gw: int, squad: pd.DataFrame) -> dict | None:
    """Distribution for the chosen XI, not just its mean.

    The mean is what the solver maximises; the shape is what decides whether
    a gameweek is survivable. Returns None rather than failing the run, since
    the brief is still useful without it.
    """
    try:
        from .engine import expected_points
        from .simulate import simulate_xi, summarise
        from .teams import fixture_expectations, team_strength

        strength = team_strength(players, teams)
        fx = fixture_expectations(strength, fixtures, gw)
        detail = expected_points(players, fx, strength, detail=True)

        xi = squad[squad["starting"]]["id"].tolist()
        captain = int(squad[squad["captain"]]["id"].iloc[0])
        if not set(xi) <= set(detail["id"]):
            return None
        totals, _ = simulate_xi(detail, xi, captain, n_sims=20_000, seed=7)
        return summarise(totals, "xi")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: simulation skipped ({exc})", file=sys.stderr)
        return None


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
                 f"{r.get('obj', r.get('horizon', 0)):.1f} | "
                 f"{r['reduced_cost']:.2f} | {r['selected_by_percent']}% |")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
