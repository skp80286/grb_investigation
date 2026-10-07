#!/usr/bin/env python3

import argparse
import sys
import pandas as pd


COLUMNS = [
    "Times",
    "Filt",
    "Freqs",
    "Fluxes",
    "FluxErrs",
    "UL",
    "Instrument",
]


def main():

    parser = argparse.ArgumentParser(
        description="Extract and reorder GRB light-curve CSV columns."
    )

    parser.add_argument("input", help="Input CSV file")

    parser.add_argument(
        "-o", "--output", default=None, help="Output CSV file (default: stdout)"
    )

    args = parser.parse_args()

    # Read input CSV
    df = pd.read_csv(args.input)

    # Check required columns
    missing = [col for col in COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Select columns in required order
    output = df[COLUMNS].copy()

    # Write to file or stdout
    if args.output:
        output.to_csv(args.output, index=False)
    else:
        output.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
