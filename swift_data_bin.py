#!/usr/bin/env python3

"""
Bin Swift-XRT data in a GRB afterglow light curve.

Input columns:
    Times,Filt,Freqs,Fluxes,FluxErrs,UL,Instrument

Behavior:
    - XRT detections are binned logarithmically in time.
    - Optical/radio/other data are left unchanged.
    - XRT fluxes are inverse-variance weighted.
    - Statistical errors are propagated.
    - Optional fractional systematic error is added in quadrature.
    - XRT upper limits are not combined with detections.

Output:
    Same CSV format as input.
"""

import argparse
import numpy as np
import pandas as pd


def weighted_bin(times, fluxes, errors, systematic=0.0):
    """
    Calculate an inverse-variance weighted mean.

    Parameters
    ----------
    times : array
        Observation times.

    fluxes : array
        Flux densities.

    errors : array
        Statistical flux errors.

    systematic : float
        Fractional systematic uncertainty.

    Returns
    -------
    time : float
        Geometric mean time.

    flux : float
        Weighted mean flux.

    error : float
        Error on weighted mean including systematic floor.
    """

    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    # Remove invalid values
    valid = (
        np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(errors) & (errors > 0)
    )

    times = times[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]

    if len(times) == 0:
        return None

    # Inverse variance weights
    weights = 1.0 / errors**2

    flux = np.sum(weights * fluxes) / np.sum(weights)

    # Statistical error on weighted mean
    stat_error = np.sqrt(1.0 / np.sum(weights))

    # Add fractional systematic uncertainty
    sys_error = systematic * abs(flux)

    total_error = np.sqrt(stat_error**2 + sys_error**2)

    # Geometric mean is more appropriate for logarithmic time bins
    time = np.exp(np.mean(np.log(times)))

    return time, flux, total_error


def make_log_bins(times, nbins=None, bin_width_dex=None):
    """
    Create logarithmic time bins.

    Either nbins or bin_width_dex can be specified.
    """

    tmin = np.min(times)
    tmax = np.max(times)

    log_min = np.log10(tmin)
    log_max = np.log10(tmax)

    if bin_width_dex is not None:
        edges = np.arange(log_min, log_max + bin_width_dex, bin_width_dex)

        # Make sure final edge includes latest point
        if edges[-1] < log_max:
            edges = np.append(edges, log_max)

    elif nbins is not None:
        edges = np.linspace(log_min, log_max, nbins + 1)

    else:
        raise ValueError("Specify either nbins or bin_width_dex")

    return 10.0**edges


def process_xrt(
    df,
    nbins=None,
    bin_width_dex=None,
    systematic=0.03,
):
    """
    Bin XRT detections logarithmically.
    """

    # Identify detections
    xrt = df[df["Instrument"].astype(str).str.upper().str.contains("XRT")].copy()

    other = df[~df["Instrument"].astype(str).str.upper().str.contains("XRT")].copy()

    # Convert numerical columns
    xrt["Times"] = pd.to_numeric(xrt["Times"], errors="coerce")

    xrt["Fluxes"] = pd.to_numeric(xrt["Fluxes"], errors="coerce")

    xrt["FluxErrs"] = pd.to_numeric(xrt["FluxErrs"], errors="coerce")

    # --------------------------------------------------------------
    # Separate detections and upper limits
    # --------------------------------------------------------------

    ul_mask = xrt["UL"].astype(str).str.upper().isin(["Y", "YES", "1", "TRUE"])

    xrt_ul = xrt[ul_mask].copy()
    xrt_det = xrt[~ul_mask].copy()

    # Remove invalid detections
    xrt_det = xrt_det[
        np.isfinite(xrt_det["Times"])
        & np.isfinite(xrt_det["Fluxes"])
        & np.isfinite(xrt_det["FluxErrs"])
        & (xrt_det["Times"] > 0)
        & (xrt_det["FluxErrs"] > 0)
    ].copy()

    if len(xrt_det) == 0:
        print("No XRT detections found.")
        return df

    # --------------------------------------------------------------
    # Construct logarithmic bins
    # --------------------------------------------------------------

    bin_edges = make_log_bins(
        xrt_det["Times"].values,
        nbins=nbins,
        bin_width_dex=bin_width_dex,
    )

    rows = []

    for i in range(len(bin_edges) - 1):
        tmin = bin_edges[i]
        tmax = bin_edges[i + 1]

        # Include right edge in final bin
        if i == len(bin_edges) - 2:
            mask = (xrt_det["Times"] >= tmin) & (xrt_det["Times"] <= tmax)
        else:
            mask = (xrt_det["Times"] >= tmin) & (xrt_det["Times"] < tmax)

        group = xrt_det[mask]

        if len(group) == 0:
            continue

        result = weighted_bin(
            group["Times"].values,
            group["Fluxes"].values,
            group["FluxErrs"].values,
            systematic=systematic,
        )

        if result is None:
            continue

        time, flux, error = result

        # Use metadata from first point in bin
        first = group.iloc[0]

        rows.append(
            {
                "Times": time,
                "Filt": first["Filt"],
                "Freqs": first["Freqs"],
                "Fluxes": flux,
                "FluxErrs": error,
                "UL": "N",
                "Instrument": first["Instrument"],
            }
        )

    binned = pd.DataFrame(rows)

    # --------------------------------------------------------------
    # Add upper limits back unchanged
    # --------------------------------------------------------------

    if len(xrt_ul) > 0:
        xrt_out = pd.concat([binned, xrt_ul], ignore_index=True)
    else:
        xrt_out = binned

    # --------------------------------------------------------------
    # Combine XRT with non-XRT data
    # --------------------------------------------------------------

    output = pd.concat([xrt_out, other], ignore_index=True)

    output.sort_values("Times", inplace=True)

    output.reset_index(drop=True, inplace=True)

    return output


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Logarithmically bin XRT data while leaving optical/radio data unchanged."
        )
    )

    parser.add_argument("input", help="Input GRB light-curve CSV")

    parser.add_argument(
        "-o", "--output", default="lightcurve_binned.csv", help="Output CSV"
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=40,
        help=("Number of logarithmic XRT bins (default: 40)"),
    )

    parser.add_argument(
        "--bin-width-dex",
        type=float,
        default=None,
        help=("Alternative to --nbins: logarithmic bin width in dex, e.g. 0.1"),
    )

    parser.add_argument(
        "--systematic",
        type=float,
        default=0.03,
        help=("Fractional XRT systematic error (default: 0.03 = 3%%)"),
    )

    args = parser.parse_args()

    if args.nbins is not None and args.bin_width_dex is not None:
        raise ValueError("Use either --nbins or --bin-width-dex, not both.")

    # --------------------------------------------------------------
    # Read input
    # --------------------------------------------------------------

    df = pd.read_csv(args.input)

    required_columns = [
        "Times",
        "Filt",
        "Freqs",
        "Fluxes",
        "FluxErrs",
        "UL",
        "Instrument",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Input points: {len(df)}")

    n_xrt = df["Instrument"].astype(str).str.upper().str.contains("XRT").sum()

    print(f"XRT points:   {n_xrt}")

    print(f"Other points: {len(df) - n_xrt}")

    # --------------------------------------------------------------
    # Process
    # --------------------------------------------------------------

    result = process_xrt(
        df,
        nbins=args.nbins,
        bin_width_dex=args.bin_width_dex,
        systematic=args.systematic,
    )

    # --------------------------------------------------------------
    # Preserve exact column order
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Write
    # --------------------------------------------------------------

    result.to_csv(args.output, index=False)

    n_xrt_out = result["Instrument"].astype(str).str.upper().str.contains("XRT").sum()

    print(f"\nOutput points: {len(result)}")

    print(f"XRT output:   {n_xrt_out}")

    print(f"Other output: {len(result) - n_xrt_out}")

    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
