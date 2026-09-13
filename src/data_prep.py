"""
data_prep.py — Data loading, Likert encoding, and missing-data reporting.

Reads the survey CSV, extracts statement codes from column headers,
encodes Likert responses numerically, and reports missing-data statistics.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── Likert encoding map ──────────────────────────────────────────────────────
LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree":          -1,
    "Neutral":            0,
    "Agree":              1,
    "Strongly Agree":     2,
}

# Values treated as missing (not encoded as Neutral)
MISSING_TOKENS = {"No Comments", ""}

# ── Category prefix → human-readable label ───────────────────────────────────
CATEGORY_LABELS = {
    "T": "Technology / AI",
    "E": "Education",
    "S": "Ethics / Society",
    "V": "Environment",
}


def _extract_code(col_name: str) -> str | None:
    """Return the statement code (e.g. 'T01') from a column header, or None."""
    m = re.match(r"^([TESV]\d{2})\.", col_name)
    return m.group(1) if m else None


def load_and_encode(csv_path: str | Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Load the survey CSV and encode it.

    Returns
    -------
    df_encoded : DataFrame
        Index = Response ID, columns = statement codes (T01 … V15),
        values = numeric Likert (−2 … +2), NaN for missing / No Comments.
    code_to_text : dict
        Maps each code (e.g. 'T01') to the full statement text.
    missing_report : DataFrame
        Missing-data counts per statement and per respondent.
    """
    raw = pd.read_csv(csv_path, dtype=str)

    # Identify the ID column and statement columns
    id_col = raw.columns[0]
    raw[id_col] = raw[id_col].astype(int)
    raw = raw.set_index(id_col)
    raw.index.name = "ResponseID"

    # Build code mapping
    code_to_text: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for col in raw.columns:
        code = _extract_code(col)
        if code:
            code_to_text[code] = col.split(". ", 1)[1]
            rename_map[col] = code

    raw = raw.rename(columns=rename_map)
    raw = raw[[c for c in raw.columns if re.match(r"^[TESV]\d{2}$", c)]]

    # Encode
    def _encode(val):
        if pd.isna(val):
            return np.nan
        val = val.strip()
        if val in MISSING_TOKENS:
            return np.nan
        return LIKERT_MAP.get(val, np.nan)

    df_encoded = raw.map(_encode)

    # Missing-data report
    missing_per_stmt = df_encoded.isna().sum().rename("missing_count")
    missing_per_resp = df_encoded.isna().sum(axis=1).rename("missing_count")
    missing_report = pd.DataFrame({
        "total_missing": [int(df_encoded.isna().sum().sum())],
        "total_cells": [df_encoded.shape[0] * df_encoded.shape[1]],
        "pct_missing": [round(df_encoded.isna().sum().sum() / (df_encoded.shape[0] * df_encoded.shape[1]) * 100, 2)],
    })

    return df_encoded, code_to_text, missing_report


def get_category(code: str) -> str:
    """Return the category letter ('T', 'E', 'S', 'V') for a statement code."""
    return code[0]


def get_category_label(code: str) -> str:
    """Return the human-readable category label for a statement code."""
    return CATEGORY_LABELS[code[0]]


def missing_data_details(df_encoded: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return per-statement and per-respondent missing counts."""
    per_stmt = df_encoded.isna().sum().sort_values(ascending=False)
    per_resp = df_encoded.isna().sum(axis=1).sort_values(ascending=False)
    return per_stmt, per_resp
