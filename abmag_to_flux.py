#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

AB_ZEROPOINT = 48.6
LOG10 = np.log(10.0)

def ab_to_flux(mag):
    # erg s^-1 cm^-2 Hz^-1
    return 10.0 ** (-0.4 * (mag + AB_ZEROPOINT))

def aberr_to_fluxerr(flux, mag_err):
    return flux * (0.4 * LOG10) * mag_err

def convert(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required = {"Times", "Filt", "Freqs", "AB", "AB_Err"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")

    # Compute fluxes
    flux = ab_to_flux(df["AB"].values)
    flux_err = aberr_to_fluxerr(flux, df["AB_Err"].values)

    # Start from a copy of input dataframe
    out = df.copy()

    # Remove AB columns
    out = out.drop(columns=["AB", "AB_Err"])

    # Insert new columns (preserve order: Times, Fluxes, FluxErrs, Filt, Freqs, others…)
    out.insert(out.columns.get_loc("Times") + 1, "Fluxes", flux)
    out.insert(out.columns.get_loc("Fluxes") + 1, "FluxErrs", flux_err)

    out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Convert AB magnitudes to fluxes")
    parser.add_argument("input_csv", help="Input CSV with AB magnitudes")
    parser.add_argument("output_csv", help="Output CSV with fluxes")
    args = parser.parse_args()

    convert(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
