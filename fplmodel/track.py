"""Projection log: what was predicted, what happened, and the gap.

The model's rank correlation is measured every week against the whole
league. That says nothing about whether the number handed over before the
deadline -- "expect 83" -- is honest. A model can rank well and still be
badly scaled, and the only way to find out is to write the prediction down
before the gameweek and score it afterwards.

Two columns do the work. `actual` is what the team scored. `in_band` says
whether it landed inside the simulated 10th-to-90th percentile interval;
over a season that should be true about 80% of the time. Much more often
and the intervals are too wide to be useful; much less and they are
dishonest.

    python -m fplmodel.track --log            # show the log and calibration
    python -m fplmodel.track --actual 7:62    # record what GW7 scored
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C

PATH = C.ROOT / "projections.csv"
COLS = ["gw", "projected", "p10", "median", "p90", "p_under40", "p_over70",
        "chip", "transfers", "actual", "error", "in_band", "note"]


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(PATH)


def record(gw: int, sim: dict, chip: str = "", transfers: int = 0,
           note: str = "") -> pd.DataFrame:
    """Write the pre-deadline projection for one gameweek.

    Called before the deadline, so `actual` is left blank. Re-recording the
    same gameweek overwrites the row rather than appending, because a second
    run before the deadline is a revision, not a new prediction.
    """
    df = load()
    row = {
        "gw": gw, "projected": sim["mean"], "p10": sim["p10"],
        "median": sim["median"], "p90": sim["p90"],
        "p_under40": sim["P<40"], "p_over70": sim["P>70"],
        "chip": chip, "transfers": transfers,
        "actual": pd.NA, "error": pd.NA, "in_band": pd.NA, "note": note,
    }
    df = df[df["gw"] != gw]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("gw").reset_index(drop=True)
    _save(df)
    return df


def settle(gw: int, actual: int) -> pd.DataFrame:
    """Record what a gameweek actually scored and score the prediction."""
    df = load()
    if gw not in set(df["gw"]):
        raise ValueError(f"no projection logged for GW{gw}")
    i = df.index[df["gw"] == gw][0]
    # The blank columns come back from CSV as float64, which will not hold a
    # bool, so widen them before writing rather than fighting the dtype.
    for col in ("actual", "error", "in_band"):
        df[col] = df[col].astype(object)
    df.loc[i, "actual"] = int(actual)
    df.loc[i, "error"] = round(actual - float(df.loc[i, "projected"]), 1)
    df.loc[i, "in_band"] = bool(
        float(df.loc[i, "p10"]) <= actual <= float(df.loc[i, "p90"]))
    _save(df)
    return df


def calibration(df: pd.DataFrame | None = None) -> dict:
    """Is the headline number honest, and are the intervals the right width?"""
    df = load() if df is None else df
    done = df[df["actual"].notna()].copy()
    if done.empty:
        return {"settled": 0}
    err = pd.to_numeric(done["error"], errors="coerce")
    band = done["in_band"].astype(str).str.lower().isin(["true", "1"])
    return {
        "settled": int(len(done)),
        "mean_error": round(float(err.mean()), 1),
        "mean_abs_error": round(float(err.abs().mean()), 1),
        "in_band_rate": round(float(band.mean()) * 100, 1),
        "in_band_target": 80.0,
    }


def _save(df: pd.DataFrame) -> None:
    df.reindex(columns=COLS).to_csv(PATH, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="projection log")
    ap.add_argument("--log", action="store_true", help="print the log")
    ap.add_argument("--actual", type=str, default=None,
                    help="record a result as GW:POINTS, e.g. 7:62")
    args = ap.parse_args(argv)

    if args.actual:
        gw, pts = args.actual.split(":")
        settle(int(gw), int(pts))

    df = load()
    print(df.to_string(index=False))
    print()
    cal = calibration(df)
    if cal["settled"]:
        print(f"settled gameweeks     {cal['settled']}")
        print(f"mean error            {cal['mean_error']:+.1f}  "
              f"(positive = model too pessimistic)")
        print(f"mean absolute error   {cal['mean_abs_error']:.1f}")
        print(f"inside 10-90 band     {cal['in_band_rate']:.0f}%  "
              f"(target {cal['in_band_target']:.0f}%)")
    else:
        print("nothing settled yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
