"""Pull the full daily NWE naphtha forward-curve history from Eikon.

Refreshes ``forward_curve.xlsx``'s ``curve`` sheet with one row per business
day and one column per forward delivery month (month-start date), back to
``DEFAULT_START_DATE``. Re-running this script is incremental: valid rows
already in the output file are kept, and only missing business days are
re-fetched from Eikon, so it is safe to schedule as a periodic refresh.

Must run on a machine with Refinitiv Eikon/Workspace active. Set
``EIKON_APP_KEY`` in your environment or ``cash_print/local_config.json``
instead of hardcoding credentials.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import eikon as ek

from cash_print_config import DEFAULT_START_DATE, configure_eikon, data_path, today_iso

configure_eikon(ek)

DEFAULT_CHAIN = "0#CN01F:"
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0


def fetch_curve_snapshot(date_str: str, chain: str) -> pd.Series | None:
    """Fetch one day's forward curve as a Series of {month_start: price}."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fwd_data, err = ek.get_data(
                instruments=chain,
                fields=["TRDPRC_1", "CF_DATE"],
                parameters={"SDate": date_str, "EDate": date_str},
            )
            if err:
                print(f"  {date_str}: Eikon reported errors: {err}")
            if fwd_data is None or fwd_data.empty:
                return None

            fwd_data = fwd_data.dropna(subset=["CF_DATE", "TRDPRC_1"])
            if fwd_data.empty:
                return None

            fwd_data["Month"] = pd.to_datetime(fwd_data["CF_DATE"]).values.astype("datetime64[M]")
            row = fwd_data.groupby("Month")["TRDPRC_1"].mean()
            row.name = pd.Timestamp(date_str)
            return row
        except Exception as exc:
            print(f"  {date_str}: attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    print(f"  {date_str}: giving up after {MAX_RETRIES} attempts")
    return None


def load_existing_curve(output_path: Path) -> pd.DataFrame:
    """Load previously fetched, genuinely valid curve rows for resuming."""

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
    curve.columns = pd.to_datetime(curve.columns)
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
    parser.add_argument("--chain", default=DEFAULT_CHAIN)
    parser.add_argument("--output", default=str(data_path("forward_curve.xlsx")))
    parser.add_argument(
        "--sleep", type=float, default=0.2, help="Delay between Eikon calls, in seconds."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25, help="Rows fetched between incremental saves."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch every business day even if already present in the output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    existing_curve = pd.DataFrame() if args.force_refresh else load_existing_curve(output_path)
    nis = load_existing_nis(output_path)

    business_days = pd.bdate_range(args.start_date, args.end_date)
    if not existing_curve.empty:
        already_have = set(existing_curve.index.normalize())
        to_fetch = [d for d in business_days if d.normalize() not in already_have]
    else:
        to_fetch = list(business_days)

    print(
        f"{len(business_days)} business days in range; {len(to_fetch)} need fetching "
        f"({len(business_days) - len(to_fetch)} already present in {output_path.name})."
    )

    curve = existing_curve.copy()
    pending_rows: list[pd.Series] = []
    for i, day in enumerate(to_fetch, start=1):
        date_str = day.strftime("%Y-%m-%d")
        print(f"[{i}/{len(to_fetch)}] Fetching {date_str}")
        row = fetch_curve_snapshot(date_str, args.chain)
        if row is not None:
            pending_rows.append(row)

        if args.sleep:
            time.sleep(args.sleep)

        if pending_rows and i % args.checkpoint_every == 0:
            curve = pd.concat([curve, pd.DataFrame(pending_rows)], axis=0)
            curve = curve[~curve.index.duplicated(keep="last")]
            pending_rows = []
            save(curve, nis, output_path)
            print(f"  checkpoint saved: {len(curve)} total rows")

    if pending_rows:
        curve = pd.concat([curve, pd.DataFrame(pending_rows)], axis=0)
        curve = curve[~curve.index.duplicated(keep="last")]

    save(curve, nis, output_path)
    print(f"\nSaved {len(curve)} rows ({curve.index.min()} to {curve.index.max()}) to {output_path}")


if __name__ == "__main__":
    main()
