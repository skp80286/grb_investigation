#!/usr/bin/env python3
"""Read a CSV and rewrite Freqs, Fluxes, FluxErrs as scientific notation (X.XXE±YY)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCI_COLUMNS = ("Freqs", "Fluxes", "FluxErrs")
DROP_COLUMNS = ("Instrument",)
TIMES_COLUMN = "Times"
FILT_COLUMN = "Filt"


def format_scientific(value: str) -> str:
    """Convert a cell to #.##E# style; pass through empty or non-numeric unchanged."""
    s = value.strip()
    if not s:
        return value
    try:
        x = float(s)
    except ValueError:
        return value
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.2E}"


def format_times(value: str) -> str:
    """Drop insignificant digits (trailing zeros, float noise) while keeping numeric value."""
    s = value.strip()
    if not s:
        return value
    low = s.lower()
    if "e" in low:
        try:
            x = float(s)
        except ValueError:
            return value
        return _format_float_compact(x)
    try:
        d = Decimal(s)
    except InvalidOperation:
        try:
            x = float(s)
        except ValueError:
            return value
        return _format_float_compact(x)
    if d.is_nan():
        return "nan"
    if not d.is_finite():
        return "-inf" if d.is_signed() else "inf"
    d = d.normalize()
    out = format(d, "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out


def _format_float_compact(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    t = f"{x:.15g}"
    if "e" in t.lower():
        return t
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Load a CSV and write a new CSV where Freqs, Fluxes, and FluxErrs "
            "use scientific notation with a two-decimal mantissa (e.g. 2.42E+18). "
            "Times values are written without insignificant trailing digits. "
            "Rows whose Filt value contains a single quote (') are dropped. "
            "The Instrument column is omitted from the output."
        )
    )
    p.add_argument("input", type=Path, help="Input CSV path")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout)",
    )
    args = p.parse_args()

    if not args.input.is_file():
        sys.stderr.write(f"error: input not found: {args.input}\n")
        sys.exit(1)

    with args.input.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            sys.stderr.write("error: CSV has no header row\n")
            sys.exit(1)
        missing = [c for c in SCI_COLUMNS if c not in reader.fieldnames]
        if missing:
            sys.stderr.write(
                f"error: missing column(s) {missing}; have {list(reader.fieldnames)}\n"
            )
            sys.exit(1)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if FILT_COLUMN in fieldnames:
        rows = [r for r in rows if "'" not in (r.get(FILT_COLUMN) or "")]

    fieldnames_out = [f for f in fieldnames if f not in DROP_COLUMNS]

    for row in rows:
        for col in SCI_COLUMNS:
            row[col] = format_scientific(row[col])
        if TIMES_COLUMN in row:
            row[TIMES_COLUMN] = format_times(row[TIMES_COLUMN])

    out = args.output
    if out is None:
        writer = csv.DictWriter(
            sys.stdout, fieldnames=fieldnames_out, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    else:
        with out.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(
                f_out, fieldnames=fieldnames_out, lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
