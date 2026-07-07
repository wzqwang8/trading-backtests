"""Shared configuration helpers for the cash diff research scripts.

The legacy notebooks in this folder were originally run from a desk-specific
Windows path with a hardcoded Eikon key. These helpers keep the scripts portable:

- local inputs/outputs resolve relative to this folder;
- Eikon credentials come from the ``EIKON_APP_KEY`` environment variable;
- end dates can default to today's date when refreshing the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_START_DATE = "2022-01-07"


def data_path(filename: str) -> Path:
    """Return a path inside the cash_print project folder."""

    return DATA_DIR / filename


def today_iso() -> str:
    """Return today's date in the yyyy-mm-dd format expected by Eikon."""

    return pd.Timestamp.today(tz=None).normalize().strftime("%Y-%m-%d")


def configure_eikon(ek_module) -> None:
    """Configure Eikon if ``EIKON_APP_KEY`` is available.

    Eikon Desktop/Workspace still needs to be running locally. Without the env
    var, scripts may still work if the user's Eikon session is already
    authenticated, but we avoid committing credentials to source control.
    """

    app_key = os.getenv("EIKON_APP_KEY")
    if app_key:
        ek_module.set_app_key(app_key)
    else:
        print(
            "EIKON_APP_KEY is not set. If Eikon calls fail, export your app key "
            "before running this script."
        )
