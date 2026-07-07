"""Shared configuration helpers for the cash diff research scripts.

The legacy notebooks in this folder were originally run from a desk-specific
Windows path with a hardcoded Eikon key. These helpers keep the scripts portable:

- local inputs/outputs resolve relative to this folder;
- Eikon credentials come from ``EIKON_APP_KEY`` or gitignored
  ``local_config.json``;
- end dates can default to today's date when refreshing the dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = DATA_DIR / "local_config.json"
DEFAULT_START_DATE = "2022-01-07"


def data_path(filename: str) -> Path:
    """Return a path inside the cash_print project folder."""

    return DATA_DIR / filename


def today_iso() -> str:
    """Return today's date in the yyyy-mm-dd format expected by Eikon."""

    return pd.Timestamp.today(tz=None).normalize().strftime("%Y-%m-%d")


def load_local_config() -> dict[str, str]:
    """Load optional gitignored local settings."""

    if not LOCAL_CONFIG_PATH.exists():
        return {}

    try:
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Could not read {LOCAL_CONFIG_PATH}: {exc}") from exc

    if not isinstance(config, dict):
        raise RuntimeError(f"{LOCAL_CONFIG_PATH} must contain a JSON object.")

    return config


def get_eikon_app_key() -> str | None:
    """Return Eikon app key from env var first, then local_config.json."""

    return os.getenv("EIKON_APP_KEY") or load_local_config().get("EIKON_APP_KEY")


def configure_eikon(ek_module) -> None:
    """Configure Eikon if a local key is available.

    Eikon Desktop/Workspace still needs to be running locally. Without a key,
    scripts may still work if the user's Eikon session is already authenticated,
    but we avoid committing credentials to source control.
    """

    app_key = get_eikon_app_key()
    if app_key:
        ek_module.set_app_key(app_key)
    else:
        print(
            "No Eikon app key found. If Eikon calls fail, set EIKON_APP_KEY or "
            f"create {LOCAL_CONFIG_PATH}."
        )
