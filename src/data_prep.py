"""
data_prep.py — Data loading, Likert encoding, missing-data audit and respondent filtering.

Reads the survey CSV, extracts statement codes from column headers, encodes
Likert responses numerically and separates two kinds of missingness:
  * blank cells   — respondent abandoned the form (dropout)
  * "No Comments" — respondent deliberately skipped a single statement
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Likert encoding map ──────────────────────────────────────────────────────
LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree":          -1,
    "Neutral":            0,
    "Agree":              1,
    "Strongly Agree":     2,
}

NO_COMMENT_TOKEN = "No Comments"

# ── Category prefix → human-readable label ───────────────────────────────────
CATEGORY_LABELS = {
    "T": "Technology / AI",
    "E": "Education",
    "S": "Ethics / Society",
    "V": "Environment",
}
DOMAINS = list(CATEGORY_LABELS)


def _extract_code(col_name: str) -> str | None:
    """Return the statement code (e.g. 'T01') from a column header, or None."""
    m = re.match(r"^([TESV]\d{2})\.", col_name)
    return m.group(1) if m else None


def load_and_encode(csv_path: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load the survey CSV and encode it.

    Returns
    -------
    df_encoded : DataFrame
        Index = Response ID, columns = statement codes (T01 … V15),
        values = numeric Likert (−2 … +2), NaN for blank / No Comments.
    code_to_text : dict
        Maps each code (e.g. 'T01') to the full statement text.
    missing : dict
        Counts of blank (dropout) and "No Comments" cells.
    """
    raw = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    id_col = raw.columns[0]
    raw[id_col] = raw[id_col].astype(int)
    raw = raw.set_index(id_col)
    raw.index.name = "ResponseID"

    code_to_text, rename_map = {}, {}
    for col in raw.columns:
        code = _extract_code(col)
        if code:
            code_to_text[code] = col.split(". ", 1)[1]
            rename_map[col] = code
    raw = raw.rename(columns=rename_map)[list(rename_map.values())]
    raw = raw.apply(lambda s: s.str.strip())
    unknown = set(raw.stack().dropna()) - set(LIKERT_MAP) - {NO_COMMENT_TOKEN}
    if unknown:
        raise ValueError(f"Unexpected response values: {sorted(unknown)}")

    df_encoded = raw.apply(lambda s: s.map(LIKERT_MAP)).astype(float)

    missing = {
        "total_cells": int(raw.size),
        "blank": int(raw.isna().sum().sum()),
        "no_comments": int((raw == NO_COMMENT_TOKEN).sum().sum()),
    }
    missing["total_missing"] = missing["blank"] + missing["no_comments"]
    return df_encoded, code_to_text, missing


def filter_respondents(df: pd.DataFrame, min_answered: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop respondents who answered fewer than `min_answered` statements.

    Their similarity to others would rest on too few (or zero) shared items,
    producing isolated nodes and spurious communities.

    Returns the kept data and a table describing the dropped respondents.
    """
    answered = df.notna().sum(axis=1)
    dropped = df.loc[answered < min_answered]
    info = pd.DataFrame({
        "answered": answered[dropped.index],
        "domains_answered": [
            "".join(d for d in DOMAINS if dropped.loc[r, [c for c in df if c[0] == d]].notna().any()) or "none"
            for r in dropped.index
        ],
    })
    return df.loc[answered >= min_answered], info


def response_style_flags(df: pd.DataFrame, straightline_share: float = 0.75) -> pd.DataFrame:
    """
    Flag unusual response styles (reported, not removed):
      * straight-lining — one answer option used for ≥ `straightline_share` of items
      * net dissenter   — mean response below zero in a strongly agreeing class
      * mostly neutral  — Neutral is the most common answer
    """
    rows = []
    for r, row in df.iterrows():
        vals = row.dropna()
        counts = vals.value_counts()
        top_val, top_share = counts.index[0], counts.iloc[0] / len(vals)
        flags = []
        if top_share >= straightline_share:
            flags.append(f"straight-lining ({top_share:.0%} same answer)")
        if vals.mean() < 0:
            flags.append(f"net dissenter (mean {vals.mean():+.2f})")
        if top_val == 0:
            flags.append(f"mostly neutral ({top_share:.0%} Neutral)")
        if flags:
            rows.append({"respondent": r, "flags": "; ".join(flags)})
    return pd.DataFrame(rows, columns=["respondent", "flags"])


def domain_columns(df: pd.DataFrame, domain: str) -> list[str]:
    """Statement columns belonging to a domain letter."""
    return [c for c in df.columns if c[0] == domain]


def get_category(code: str) -> str:
    """Return the category letter ('T', 'E', 'S', 'V') for a statement code."""
    return code[0]
