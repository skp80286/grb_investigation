#!/usr/bin/env python3

"""
Bin a multi-filter light curve in logarithmic time.

Required columns:
    Times, Filt, Freqs, Fluxes, FluxErrs

Optional columns:
    UL           assumed N when the column is absent
    Instrument   left blank when the column is absent

Behavior:
    - Each selected filter is binned on its own logarithmic time grid.
    - Detections of the same filter that fall in one time bin are combined.
    - With --filters, only those filters are binned. Other rows are copied through.
    - Fluxes are inverse-variance weighted. Errors are propagated.
    - An optional fractional systematic error is added in quadrature.
    - Upper limits are not combined with detections.
    - A filter must have one Freqs value. Differing values stop the program.
    - Combined rows with more than one instrument are labeled "Multiple".

Output columns:
    Times, Filt, Freqs, Fluxes, FluxErrs, UL, Instrument

Printed to stdout unless --output is set.
"""

import argparse
import sys

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "Times",
    "Filt",
    "Freqs",
    "Fluxes",
    "FluxErrs",
    "UL",
    "Instrument",
]

REQUIRED_COLUMNS = ["Times", "Filt", "Freqs", "Fluxes", "FluxErrs"]

UL_TRUE = {"Y", "YES", "1", "TRUE"}


def weighted_bin(times, fluxes, errors, systematic=0.0):
    """Inverse-variance weighted mean at the geometric-mean time."""

    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = (
        np.isfinite(times)
        & np.isfinite(fluxes)
        & np.isfinite(errors)
        & (times > 0)
        & (errors > 0)
    )

    times = times[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]

    if len(times) == 0:
        return None

    weights = 1.0 / errors**2
    flux = np.sum(weights * fluxes) / np.sum(weights)
    stat_error = np.sqrt(1.0 / np.sum(weights))
    sys_error = systematic * abs(flux)
    total_error = np.sqrt(stat_error**2 + sys_error**2)
    time = np.exp(np.mean(np.log(times)))

    return time, flux, total_error


def make_log_bins(times, nbins=None, bin_width_dex=None):
    """Logarithmic bin edges. Specify nbins or bin_width_dex."""

    tmin = np.min(times)
    tmax = np.max(times)

    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmin <= 0 or tmax <= 0:
        raise ValueError("Times must be positive and finite to build log bins")

    if np.isclose(tmin, tmax):
        return np.array([tmin, np.nextafter(tmax, np.inf)])

    log_min = np.log10(tmin)
    log_max = np.log10(tmax)

    if bin_width_dex is not None:
        if bin_width_dex <= 0:
            raise ValueError("--bin-width-dex must be positive")
        edges = np.arange(log_min, log_max + bin_width_dex, bin_width_dex)
        if edges[-1] < log_max:
            edges = np.append(edges, log_max)
    elif nbins is not None:
        if nbins < 1:
            raise ValueError("--nbins must be at least 1")
        edges = np.linspace(log_min, log_max, nbins + 1)
    else:
        raise ValueError("Specify either nbins or bin_width_dex")

    if len(edges) < 2:
        return np.array([tmin, np.nextafter(float(tmax), np.inf)])

    bin_edges = 10.0**edges
    # 10**log10(t) can sit a fraction below the data and drop an endpoint.
    bin_edges[0] = min(bin_edges[0], float(tmin))
    bin_edges[-1] = max(bin_edges[-1], float(tmax))
    return bin_edges


def _clean_instrument(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _grouped_instrument(values):
    labels = []
    for value in values:
        text = _clean_instrument(value)
        if text not in labels:
            labels.append(text)
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return "Multiple"


def _is_upper_limit(value):
    if pd.isna(value):
        return False
    return str(value).strip().upper() in UL_TRUE


def _distinct_freqs(values):
    freqs = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = freqs.to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Freqs contains missing or non-numeric values")

    unique = []
    for value in numeric:
        if not any(np.isclose(value, kept, rtol=1e-6, atol=0.0) for kept in unique):
            unique.append(float(value))
    return unique


def _require_single_freq(filt, values):
    unique = _distinct_freqs(values)
    if len(unique) > 1:
        shown = ", ".join(f"{value:.6g}" for value in unique)
        raise ValueError(
            f"filter {filt!r} has different Freqs values: {shown}"
        )
    return unique[0]


def _prepare(df):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.copy()

    if out["Filt"].isna().any():
        raise ValueError("Filt contains missing values")
    out["Filt"] = out["Filt"].astype(str).str.strip()
    if (out["Filt"] == "").any():
        raise ValueError("Filt contains blank values")

    if "UL" not in out.columns:
        out["UL"] = "N"
    if "Instrument" not in out.columns:
        out["Instrument"] = ""

    out["Times"] = pd.to_numeric(out["Times"], errors="coerce")
    if out["Times"].isna().any() or (out["Times"] <= 0).any():
        raise ValueError("Times must be positive numbers")

    return out


def _filters_to_bin(prepared, filters):
    """Return the filter names to bin. None means every filter in the file."""

    if filters is None:
        return None
    if isinstance(filters, str):
        filters = [filters]

    selected = []
    for item in filters:
        for part in str(item).split(","):
            name = part.strip()
            if name and name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError("no filters given to --filters")

    present = list(dict.fromkeys(prepared["Filt"].tolist()))
    missing = [name for name in selected if name not in present]
    if missing:
        available = ", ".join(present)
        raise ValueError(
            f"filters not in the file: {', '.join(missing)}. Available: {available}"
        )
    return selected


def _aggregate(group, systematic):
    result = weighted_bin(
        group["Times"].values,
        group["Fluxes"].values,
        group["FluxErrs"].values,
        systematic=systematic,
    )
    if result is None:
        return None

    time, flux, error = result
    freq = _require_single_freq(group["Filt"].iloc[0], group["Freqs"])

    return {
        "Times": time,
        "Filt": group["Filt"].iloc[0],
        "Freqs": freq,
        "Fluxes": flux,
        "FluxErrs": error,
        "UL": "N",
        "Instrument": _grouped_instrument(group["Instrument"]),
    }


def _bin_filter(group, nbins, bin_width_dex, systematic):
    times = group["Times"].to_numpy(dtype=float)
    edges = make_log_bins(times, nbins=nbins, bin_width_dex=bin_width_dex)
    rows = []

    for i in range(len(edges) - 1):
        tmin = edges[i]
        tmax = edges[i + 1]
        if i == len(edges) - 2:
            mask = (group["Times"] >= tmin) & (group["Times"] <= tmax)
        else:
            mask = (group["Times"] >= tmin) & (group["Times"] < tmax)

        chunk = group[mask]
        if len(chunk) == 0:
            continue

        row = _aggregate(chunk, systematic)
        if row is not None:
            rows.append(row)

    return rows


def _passthrough(group):
    rows = []
    for _, row in group.iterrows():
        rows.append(
            {
                "Times": float(row["Times"]),
                "Filt": row["Filt"],
                "Freqs": float(pd.to_numeric(row["Freqs"], errors="coerce")),
                "Fluxes": pd.to_numeric(row["Fluxes"], errors="coerce"),
                "FluxErrs": pd.to_numeric(row["FluxErrs"], errors="coerce"),
                "UL": row["UL"] if not pd.isna(row["UL"]) else "N",
                "Instrument": _clean_instrument(row["Instrument"]),
            }
        )
    return rows


def bin_lightcurve(
    df,
    nbins=None,
    bin_width_dex=None,
    systematic=0.0,
    filters=None,
):
    """
    Bin detections of each selected filter.

    Upper limits, and every row of a filter that was not selected, are copied
    through unchanged. filters=None bins every filter.
    """

    prepared = _prepare(df)
    selected = _filters_to_bin(prepared, filters)
    if selected is None:
        selected_mask = pd.Series(True, index=prepared.index)
        bands = prepared
    else:
        selected_mask = prepared["Filt"].isin(selected)
        bands = prepared.loc[selected_mask]

    for filt, band in bands.groupby("Filt", sort=False):
        _require_single_freq(filt, band["Freqs"])

    ul_mask = prepared["UL"].map(_is_upper_limit)
    detections = prepared.loc[selected_mask & ~ul_mask].copy()
    unchanged = prepared.loc[~selected_mask | ul_mask].copy()

    if len(detections):
        detections["Fluxes"] = pd.to_numeric(detections["Fluxes"], errors="coerce")
        detections["FluxErrs"] = pd.to_numeric(detections["FluxErrs"], errors="coerce")
        invalid = (
            ~np.isfinite(detections["Fluxes"])
            | ~np.isfinite(detections["FluxErrs"])
            | (detections["FluxErrs"] <= 0)
        )
        if invalid.any():
            raise ValueError(
                f"{int(invalid.sum())} detections have missing flux or a non-positive error"
            )

    rows = []
    for _, group in detections.groupby("Filt", sort=False):
        rows.extend(
            _bin_filter(
                group,
                nbins=nbins,
                bin_width_dex=bin_width_dex,
                systematic=systematic,
            )
        )
    rows.extend(_passthrough(unchanged))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output.sort_values(["Times", "Filt"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Logarithmically bin a light curve. Points of the same filter "
            "inside a time bin are combined. Other filters are kept separate."
        )
    )

    parser.add_argument("input", help="Input light-curve CSV")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV file (default: print to stdout)",
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=None,
        help="Logarithmic bins per filter (default: 40 if --bin-width-dex is not set)",
    )

    parser.add_argument(
        "--bin-width-dex",
        type=float,
        default=None,
        help="Logarithmic bin width in dex, used instead of --nbins",
    )

    parser.add_argument(
        "--systematic",
        type=float,
        default=0.0,
        help=(
            "Fractional systematic error added in quadrature "
            "(default: 0; 0.03 is a typical XRT floor)"
        ),
    )

    parser.add_argument(
        "--filters",
        nargs="+",
        default=None,
        metavar="FILT",
        help=(
            "Filters to bin, for example: --filters r g or --filters r,g. "
            "Other filters are copied unchanged. Default: all filters."
        ),
    )

    args = parser.parse_args()

    if args.nbins is not None and args.bin_width_dex is not None:
        raise SystemExit("error: use either --nbins or --bin-width-dex, not both")

    nbins = args.nbins
    if nbins is None and args.bin_width_dex is None:
        nbins = 40

    df = pd.read_csv(args.input)

    try:
        result = bin_lightcurve(
            df,
            nbins=nbins,
            bin_width_dex=args.bin_width_dex,
            systematic=args.systematic,
            filters=args.filters,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    n_filters = df["Filt"].nunique() if "Filt" in df.columns else 0
    if args.filters:
        binned = ", ".join(args.filters)
    else:
        binned = "all"
    print(f"input points:  {len(df)}", file=sys.stderr)
    print(f"filters:       {n_filters}", file=sys.stderr)
    print(f"binning:       {binned}", file=sys.stderr)
    print(f"output points: {len(result)}", file=sys.stderr)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"written to:    {args.output}", file=sys.stderr)
    else:
        result.to_csv(sys.stdout, index=False)
        sys.stdout.flush()

    _print_band_summary(df, result)


def _print_band_summary(source, result):
    """Print rows per filter before and after binning."""

    before_names = source["Filt"].astype(str).str.strip()
    before_counts = before_names.value_counts(sort=False)
    after_counts = result["Filt"].astype(str).value_counts(sort=False)

    order = list(dict.fromkeys(before_names.tolist()))
    for name in after_counts.index:
        if name not in order:
            order.append(name)

    print("band, rows before, rows after", file=sys.stderr)
    for name in order:
        before = int(before_counts.get(name, 0))
        after = int(after_counts.get(name, 0))
        print(f"{name}, {before}, {after}", file=sys.stderr)


if __name__ == "__main__":
    main()
