"""Pull the full daily NWE naphtha forward-curve history from Eikon.

Uses the rolling monthly-contract RICs (Mo01..Mo12) via ``ek.get_timeseries``
instead of a historical chain snapshot. Mo01 is always the current
front-month naphtha swap price, Mo02 the next month, and so on, so one bulk
time-series call per contract yields the full curve history in a single
shot per RIC. This replaces an earlier day-by-day chain-snapshot approach
(``0#NAF-NWE:``) that returned "record could not be found" for anything in
roughly the last several months, most likely because that chain does not
support historical point-in-time snapshots that recently.

Refreshes ``forward_curve.xlsx``'s ``curve`` sheet with one row per business
day and one column per month-forward offset (1 = front month, 2 = next
month, ... 12 = twelve months out).

Must run on a machine with Refinitiv Eikon/Workspace active. Set
``EIKON_APP_KEY`` in your environment or ``cash_print/local_config.json``
instead of hardcoding credentials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import eikon as ek

from cash_print_config import DEFAULT_START_DATE, configure_eikon, data_path, today_iso

configure_eikon(ek)

# Naphtha CIF NWE Cargo Financial rolling monthly contracts: Mo01 (front
# month) through Mo12. Mo01 is the same RIC used as "MOC" elsewhere in this
# project (PAAAJ00).
MONTHLY_CONTRACT_RICS = {
    1: "PAAAJ00",
    2: "AAECO00",
    3: "AAECQ00",
    4: "AAECR00",
    5: "AAEN005",
    6: "AAEN006",
    7: "AAEN007",
    8: "AAEN008",
    9: "AAEN009",
    10: "AAEN010",
    11: "AAEN011",
    12: "AAEN012",
}


def fetch_curve_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Pull each rolling monthly contract's daily close in one shot each."""

    columns = {}
    for month_offset, ric in MONTHLY_CONTRACT_RICS.items():
        try:
            data = ek.get_timeseries(ric, start_date=start_date, end_date=end_date)["CLOSE"]
            columns[month_offset] = data
            print(f"  Mo{month_offset:02d} ({ric}): {len(data)} rows")
        except Exception as exc:
            print(f"  Mo{month_offset:02d} ({ric}): failed: {exc}")

    curve = pd.DataFrame(columns)
    curve.index.name = "Date"
    return curve.sort_index()


def load_existing_curve(output_path: Path) -> pd.DataFrame:
    """Load previously saved, genuinely valid curve rows to merge with."""

    if not output_path.exists():
        return pd.DataFrame()
    try:
        existing = pd.read_excel(output_path, sheet_name="curve")
    except Exception:
        return pd.DataFrame()

    date_col = "Date" if "Date" in existing.columns else existing.columns[0]
    existing[date_col] = pd.to_datetime(existing[date_col], errors="coerce")
    existing = existing.dropna(subset=[date_col]).set_index(date_col)
    existing = existing[~existing.index.duplicated(keep="last")]
    # Drop legacy formula-based placeholder rows that were all zero.
    existing = existing.loc[(existing.fillna(0) != 0).any(axis=1)]

    # Only reuse rows that already use the month-forward-offset column
    # schema (1..12). Older calendar-date-column snapshots use a different,
    # incompatible schema; concatenating the two would leave every row with
    # data under only its own column set and NaN under the other's, so
    # discard incompatible legacy data instead of merging with it.
    try:
        existing.columns = existing.columns.astype(float).astype(int)
    except (TypeError, ValueError):
        return pd.DataFrame()
    return existing


def load_existing_nis(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists():
        return None
    try:
        return pd.read_excel(output_path, sheet_name="NIS")
    except Exception:
        return None


def save(curve: pd.DataFrame, nis: pd.DataFrame | None, output_path: Path) -> None:
    curve = curve.sort_index()
    curve = curve.reindex(sorted(curve.columns), axis=1)
    curve.index.name = "Date"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        curve.to_excel(writer, sheet_name="curve")
        if nis is not None:
            nis.to_excel(writer, sheet_name="NIS", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=today_iso())
    parser.add_argument("--output", default=str(data_path("forward_curve.xlsx")))
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore any existing output file and rebuild it from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    existing_curve = pd.DataFrame() if args.force_refresh else load_existing_curve(output_path)
    nis = load_existing_nis(output_path)

    print(f"Fetching Mo01-Mo12 curve history from {args.start_date} to {args.end_date}...")
    fresh = fetch_curve_history(args.start_date, args.end_date)

    if not existing_curve.empty:
        curve = pd.concat([existing_curve, fresh], axis=0)
        curve = curve[~curve.index.duplicated(keep="last")]
    else:
        curve = fresh

    save(curve, nis, output_path)
    print(f"\nSaved {len(curve)} rows ({curve.index.min()} to {curve.index.max()}) to {output_path}")


if __name__ == "__main__":
    main()
