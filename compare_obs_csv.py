#!/usr/bin/env python3
"""Compare two observation CSVs on Times, Freqs, Fluxes, FluxErrs, Filt.

Rows are matched on (Times, Filt), with duplicate keys paired by sorting within each
group on Freqs, Fluxes, FluxErrs (stable order).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = ("Times", "Freqs", "Fluxes", "FluxErrs", "Filt")
COMPARE_COLS = ("Freqs", "Fluxes", "FluxErrs")


def _stdout_use_color() -> bool:
    return sys.stdout.isatty()


def _color(code: str, text: str, use_color: bool) -> str:
    if not use_color:
        return text
    reset = "\033[0m"
    return f"{code}{text}{reset}"


def rel_pct_diff(a: float, b: float, eps: float = 1e-300) -> float | None:
    """Symmetric-ish relative difference as a fraction (not percent). Returns None if NaN."""
    if math.isnan(a) or math.isnan(b):
        return None
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b), eps)
    return abs(a - b) / denom


def augment_for_merge(df: pd.DataFrame, time_round_decimals: int) -> pd.DataFrame:
    """Add merge key columns: rounded time + occurrence index within (time_key, Filt)."""
    out = df.copy()
    out["_tkey"] = out["Times"].astype(float).round(time_round_decimals)
    out["Filt"] = out["Filt"].astype(str).str.strip()
    sort_cols = ["_tkey", "Filt"] + list(COMPARE_COLS)
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out["_occ"] = out.groupby(["_tkey", "Filt"], sort=False).cumcount()
    return out


def compare_obs_csv(
    path_a: Path,
    path_b: Path,
    *,
    threshold: float,
    time_round_decimals: int,
    label_a: str,
    label_b: str,
    use_color: bool,
) -> int:
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    for name, df in ((label_a, df_a), (label_b, df_b)):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"error: {name} missing columns: {missing}", file=sys.stderr)
            return 2

    a_aug = augment_for_merge(df_a, time_round_decimals)
    b_aug = augment_for_merge(df_b, time_round_decimals)

    merged = pd.merge(
        a_aug,
        b_aug,
        on=["_tkey", "Filt", "_occ"],
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
    )

    only_a = merged[merged["_merge"] == "left_only"].copy()
    only_b = merged[merged["_merge"] == "right_only"].copy()
    both = merged[merged["_merge"] == "both"].copy()

    hi = _color("\033[1;31m", ">", use_color)

    print(f"Compared:\n  A ({label_a}): {path_a}\n  B ({label_b}): {path_b}")
    print(f"Match key: rounded Times ({time_round_decimals} dp) + Filt + row index within duplicates")
    print(f"Difference threshold: {threshold:.4%} relative on {list(COMPARE_COLS)}\n")

    # --- Rows present in only one file ---
    def fmt_missing(df_side: pd.DataFrame, title: str, suffix: str) -> None:
        if df_side.empty:
            print(f"{title}: none.")
            return
        cols_show = ["Times", "Freqs", "Fluxes", "FluxErrs", "Filt"]
        renamed = pd.DataFrame()
        for c in cols_show:
            if c == "Filt" and "Filt" in df_side.columns:
                renamed[c] = df_side["Filt"]
            elif f"{c}_{suffix}" in df_side.columns:
                renamed[c] = df_side[f"{c}_{suffix}"]
        print(f"{title} ({len(df_side)} rows):")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", None)
        print(renamed.to_string(index=False))
        print()

    print("=== Rows present in only one file ===")
    fmt_missing(
        only_a,
        f"Only in {label_a} (missing from {label_b})",
        "a",
    )
    fmt_missing(
        only_b,
        f"Only in {label_b} (missing from {label_a})",
        "b",
    )

    # --- Rows in both: flag columns differing by > threshold ---
    mismatch_rows: list[dict] = []
    for _, row in both.iterrows():
        issues: list[str] = []
        max_frac = 0.0
        for col in COMPARE_COLS:
            va = float(row[f"{col}_a"])
            vb = float(row[f"{col}_b"])
            frac = rel_pct_diff(va, vb)
            if frac is None:
                issues.append(f"{col}: NaN mismatch ({va} vs {vb})")
                max_frac = math.inf
            elif frac > threshold:
                pct = 100.0 * frac
                issues.append(f"{col}: {va} vs {vb} ({pct:.4g}% diff)")
                max_frac = max(max_frac, frac)
        if issues:
            mismatch_rows.append(
                {
                    "Times_a": row["Times_a"],
                    "Times_b": row["Times_b"],
                    "Filt": row["Filt"],
                    "_occ": row["_occ"],
                    "max_rel_diff": max_frac,
                    "issues": issues,
                }
            )

    print("=== Rows present in both files with >threshold value differences ===")
    if not mismatch_rows:
        print("None.")
    else:
        mismatch_rows.sort(key=lambda r: (-(r["max_rel_diff"] or 0), str(r["Filt"]), r["_occ"]))
        for r in mismatch_rows:
            head = (
                f"{hi} Times {r['Times_a']} / {r['Times_b']} | Filt {r['Filt']} | "
                f"dup# {r['_occ']}"
            )
            print(head)
            for issue in r["issues"]:
                print(f"    {issue}")
            print()

    # Optional: tiny Time drift within matched rows
    time_drift = []
    for _, row in both.iterrows():
        ta, tb = float(row["Times_a"]), float(row["Times_b"])
        if ta != tb:
            frac = rel_pct_diff(ta, tb)
            if frac is not None and frac > threshold:
                time_drift.append((ta, tb, row["Filt"], row["_occ"], frac))

    if time_drift:
        print("=== Note: Times differ between files for the same key (beyond threshold) ===")
        for ta, tb, filt, occ, frac in sorted(time_drift, key=lambda x: -x[4]):
            print(
                f"  {hi} Filt={filt} dup#={occ}: Times {ta} vs {tb} "
                f"({100.0 * frac:.4g}% relative diff)"
            )
        print()

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file_a", type=Path, help="First CSV path")
    p.add_argument("file_b", type=Path, help="Second CSV path")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Relative difference above this value is highlighted (default: 0.01 = 1%%)",
    )
    p.add_argument(
        "--time-dp",
        type=int,
        default=6,
        help="Decimal places for rounding Times in the match key (default: 6)",
    )
    p.add_argument("--label-a", default="A", help="Label for first file in output")
    p.add_argument("--label-b", default="B", help="Label for second file in output")
    p.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Highlight with ANSI colors (default: on if stdout is a TTY)",
    )

    args = p.parse_args()
    use_color = args.color and _stdout_use_color()

    return compare_obs_csv(
        args.file_a.resolve(),
        args.file_b.resolve(),
        threshold=args.threshold,
        time_round_decimals=args.time_dp,
        label_a=args.label_a,
        label_b=args.label_b,
        use_color=use_color,
    )


if __name__ == "__main__":
    raise SystemExit(main())
