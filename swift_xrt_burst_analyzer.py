#!/usr/bin/env python3

"""
Download Swift Burst Analyser XRT flux-density data for a GRB
and convert it to:

Times,Filt,Freqs,Fluxes,FluxErrs,UL,Instrument

FluxErrs is taken as:

    max(abs(positive_error), abs(negative_error))

The XRT Density product is the unabsorbed flux density at 10 keV,
converted from Jy to mJy.
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    import swifttools.ukssdc.data.GRB as udg
except ImportError:
    print(
        "ERROR: swifttools is not installed.\n"
        "\n"
        "Install it with:\n"
        "    pip install swifttools\n"
    )
    sys.exit(1)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Burst Analyser XRT Density product is at 10 keV
E_XRT_KEV = 10.0

# Planck constant in keV s
H_KEV_S = 4.135667696e-18

# Frequency corresponding to 10 keV
FREQ_XRT_HZ = E_XRT_KEV / H_KEV_S

# Swift density is in Jy
# 1 Jy = 1000 mJy
JY_TO_MJY = 1000.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download Swift Burst Analyser XRT flux-density "
            "light curve and write it in the standard GRB CSV format."
        )
    )

    parser.add_argument(
        "--grb",
        default="GRB 260924A",
        help="GRB name (default: GRB 260924A)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="GRB260924A_XRT.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "--download-dir",
        default="swift_burst_analyser",
        help="Directory in which Swift data are downloaded",
    )

    parser.add_argument(
        "--propagated-errors",
        action="store_true",
        help=(
            "Use errors including uncertainty in the spectral "
            "conversion factor. Recommended."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Download Burst Analyser data
    # ------------------------------------------------------------------

    print(f"Downloading Swift Burst Analyser data for {args.grb}...")

    data = udg.getBurstAnalyser(
        GRBName=args.grb,
        instruments=("XRT",),
        bands=("density",),
        returnData=True,
        saveData=True,
        destDir=args.download_dir,
        silent=False,
    )

    # ------------------------------------------------------------------
    # Locate GRB data
    # ------------------------------------------------------------------

    if args.grb in data:
        grb_data = data[args.grb]
    elif "XRT" in data:
        # Some swifttools versions return the GRB data directly
        grb_data = data
    else:
        raise RuntimeError(
            f"Could not find {args.grb} in returned data.\n"
            f"Available keys: {list(data.keys())}"
        )

    if "XRT" not in grb_data:
        raise RuntimeError(
            f"XRT data were not returned.\nAvailable keys: {list(grb_data.keys())}"
        )

    xrt = grb_data["XRT"]

    # ------------------------------------------------------------------
    # Find Density light curves
    # ------------------------------------------------------------------

    density_keys = [key for key in xrt.keys() if key.lower().startswith("density")]

    if not density_keys:
        raise RuntimeError(
            f"No XRT Density light curve found.\nAvailable XRT keys: {list(xrt.keys())}"
        )

    print("\nDensity datasets found:")
    for key in density_keys:
        print(f"    {key}")

    # ------------------------------------------------------------------
    # Process all density datasets
    # ------------------------------------------------------------------

    output_rows = []

    for density_key in density_keys:
        df = xrt[density_key]

        if df is None or len(df) == 0:
            continue

        print(f"\nProcessing {density_key}: {len(df)} points")

        # --------------------------------------------------------------
        # Determine XRT observing mode
        # --------------------------------------------------------------

        key_upper = density_key.upper()

        if "WT" in key_upper:
            mode = "WT"
        elif "PC" in key_upper:
            mode = "PC"
        else:
            mode = "XRT"

        # --------------------------------------------------------------
        # Select error columns
        # --------------------------------------------------------------

        if args.propagated_errors:
            plus_col = "FluxPosWithECFErr"
            minus_col = "FluxNegWithECFErr"

            if plus_col not in df.columns or minus_col not in df.columns:
                raise RuntimeError(
                    f"Propagated error columns not available "
                    f"in {density_key}.\n"
                    f"Available columns: {list(df.columns)}"
                )

        else:
            plus_col = "FluxPos"
            minus_col = "FluxNeg"

        # --------------------------------------------------------------
        # Process individual points
        # --------------------------------------------------------------

        for _, row in df.iterrows():
            time = float(row["Time"])

            flux = float(row["Flux"])

            err_plus = float(row[plus_col])
            err_minus = float(row[minus_col])

            # ----------------------------------------------------------
            # Validate
            # ----------------------------------------------------------

            if not np.isfinite(time):
                continue

            if not np.isfinite(flux):
                continue

            if not np.isfinite(err_plus):
                continue

            if not np.isfinite(err_minus):
                continue

            # ----------------------------------------------------------
            # Convert Jy -> mJy
            # ----------------------------------------------------------

            flux_mjy = flux * JY_TO_MJY

            err_plus_mjy = err_plus * JY_TO_MJY
            err_minus_mjy = err_minus * JY_TO_MJY

            # ----------------------------------------------------------
            # Symmetric error for output
            #
            # Take the larger absolute error.
            # ----------------------------------------------------------

            flux_err_mjy = max(abs(err_plus_mjy), abs(err_minus_mjy))

            # ----------------------------------------------------------
            # Upper limit
            # ----------------------------------------------------------

            if flux <= 0:
                ul = "Y"
            else:
                ul = "N"

            # ----------------------------------------------------------
            # Output row
            # ----------------------------------------------------------

            output_rows.append(
                {
                    "Times": time,
                    "Filt": "X-ray(10keV)",
                    "Freqs": FREQ_XRT_HZ,
                    "Fluxes": flux_mjy,
                    "FluxErrs": flux_err_mjy,
                    "UL": ul,
                    "Instrument": "Swift-XRT",
                }
            )

    if not output_rows:
        raise RuntimeError("No valid XRT density points were found.")

    # ------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------

    result = pd.DataFrame(output_rows)

    # Sort chronologically
    result.sort_values(
        by="Times",
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    # Ensure exact column order
    result = result[
        [
            "Times",
            "Filt",
            "Freqs",
            "Fluxes",
            "FluxErrs",
            "UL",
            "Instrument",
        ]
    ]

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------

    result.to_csv(
        args.output,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n---------------------------------------------")
    print("Swift XRT conversion complete")
    print("---------------------------------------------")

    print(f"GRB              : {args.grb}")
    print(f"Output           : {args.output}")
    print(f"Number of points : {len(result)}")
    print(f"Energy           : {E_XRT_KEV:.1f} keV")
    print(f"Frequency        : {FREQ_XRT_HZ:.8e} Hz")

    print(
        "Errors           : "
        + (
            "max(abs(+err), abs(-err)), including ECF uncertainty"
            if args.propagated_errors
            else "max(abs(+err), abs(-err))"
        )
    )

    print("\nFirst five rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
