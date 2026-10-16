"""Spreadsheet-safe CSV exports for user-supplied and FIRMS values."""

from __future__ import annotations

import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize CSV while neutralizing formula-like text cells for spreadsheets."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        values = safe[column].astype("string")
        starts_formula = values.str.lstrip().str.startswith(FORMULA_PREFIXES, na=False)
        safe.loc[starts_formula, column] = "'" + values.loc[starts_formula].astype(str)
    return safe.to_csv(index=False).encode("utf-8-sig")
