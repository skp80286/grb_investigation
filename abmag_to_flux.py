#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

AB_ZEROPOINT = 48.6
LOG10 = np.log(10.0)
CGS_TO_JY = 1e26   # erg s^-1 cm^-2 Hz^-1 -> Jy

def ab_to_flux_jy(mag):
    """
    Convert AB magnitude to flux density in Jansky (Jy)
    """
    f_cgs = 10.0 ** (-0.4 * (mag + AB_ZEROPOINT))
    return f_cgs * CGS_TO_JY

def aberr_to_fluxerr_jy(flux_jy, mag_err):
    """
    Propagate AB magnitude error into flux error (Jy)
    """
    return flux_jy * (0.4 * LOG10) * mag_err

def convert(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required = {"Times", "Filt", "Freqs", "AB", "AB_Err"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")

    # Compute fluxes in Jy
    flux_jy = ab_to_flux_jy(df["AB"].values)
    fluxerr_jy = aberr_to_fluxerr_jy(flux_jy, df["AB_Err"].values)

    # Preserve all other columns
    out = df.copy()
    out = out.drop(columns=["AB", "AB_Err"])

    # Insert new columns next to Times
    out.insert(out.columns.get_loc("Times") + 1, "Fluxes", flux_jy)
    out.insert(out.columns.get_loc("Fluxes") + 1, "FluxErrs", fluxerr_jy)

    out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Convert AB magnitudes to fluxes (Jy)")
    parser.add_argument("input_csv", help="Input CSV with AB magnitudes")
    parser.add_argument("output_csv", help="Output CSV with fluxes in Jy")
    args = parser.parse_args()

    convert(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
