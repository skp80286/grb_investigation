from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import pandas as pd


def canonical_filter_name(filter_name: str) -> str:
    """Map instrument-specific filter labels to canonical ugrizy bands."""
    key = filter_name.strip().lower()
    alias_map = {
        "sdssu": "u",
        "ztfu": "u",
        "u": "u",
        "sdssg": "g",
        "ztfg": "g",
        "g": "g",
        "sdssr": "r",
        "ztfr": "r",
        "r": "r",
        "sdssi": "i",
        "ztfi": "i",
        "i": "i",
        "sdssz": "z",
        "ztfz": "z",
        "z": "z",
        "sdssy": "y",
        "ztfy": "y",
        "y": "y",
    }
    return alias_map.get(key, key)


def set_publication_style() -> None:
    """Apply publication-ready plotting style."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "high-vis"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def plot_lightcurve(
    csv_path: Path,
    output_path: Path,
    show: bool = True,
    diff_psf_type: str | None = None,
    reference_csv_path: Path | None = None,
    target_canonical_filter: str | None = None,
) -> None:
    required_columns = {
        "filter",
        "JD",
        "diff_PSF_type",
        "diff_psf_mag",
        "diff_psf_mag_err",
    }

    df = pd.read_csv(csv_path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    if diff_psf_type:
        df = df[df["diff_PSF_type"].astype(str) == diff_psf_type]

    df = df.dropna(subset=["filter", "JD", "diff_psf_mag", "diff_psf_mag_err"]).copy()
    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df["diff_psf_mag"] = pd.to_numeric(df["diff_psf_mag"], errors="coerce")
    df["diff_psf_mag_err"] = pd.to_numeric(df["diff_psf_mag_err"], errors="coerce")
    df = df.dropna(subset=["JD", "diff_psf_mag", "diff_psf_mag_err"])
    df["MJD"] = df["JD"] - 2400000.0
    df["canonical_filter"] = df["filter"].astype(str).map(canonical_filter_name)

    if target_canonical_filter:
        df = df[df["canonical_filter"] == target_canonical_filter]

    if df.empty:
        raise ValueError("No valid rows to plot after filtering and cleaning.")

    ref_df: pd.DataFrame | None = None
    if reference_csv_path is not None:
        ref_df = pd.read_csv(reference_csv_path)
        required_ref_columns = {"mjd", "mag", "magerr", "filter"}
        missing_ref = required_ref_columns - set(ref_df.columns)
        if missing_ref:
            raise ValueError(
                f"Missing required columns in {reference_csv_path}: {sorted(missing_ref)}"
            )
        ref_df = ref_df.dropna(subset=["mjd", "mag", "magerr", "filter"]).copy()
        ref_df["mjd"] = pd.to_numeric(ref_df["mjd"], errors="coerce")
        ref_df["mag"] = pd.to_numeric(ref_df["mag"], errors="coerce")
        ref_df["magerr"] = pd.to_numeric(ref_df["magerr"], errors="coerce")
        ref_df["canonical_filter"] = ref_df["filter"].astype(str).map(canonical_filter_name)
        ref_df = ref_df.dropna(subset=["mjd", "mag", "magerr"])
        if target_canonical_filter:
            ref_df = ref_df[ref_df["canonical_filter"] == target_canonical_filter]
        if ref_df.empty:
            ref_df = None

    set_publication_style()

    color_map = {
        "u": "#7A00CC",
        "g": "#1B9E77",
        "r": "#D95F02",
        "i": "#7570B3",
        "z": "#E7298A",
        "y": "#66A61E",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for filt in sorted(df["filter"].astype(str).unique()):
        sub = df[df["filter"].astype(str) == filt].sort_values("JD")
        color = color_map.get(canonical_filter_name(filt), None)
        psf_sub = sub[sub["diff_PSF_type"].astype(str) == "PSF"]
        forced_sub = sub[sub["diff_PSF_type"].astype(str) == "Forced_PSF"]

        if not psf_sub.empty:
            ax.errorbar(
                psf_sub["MJD"],
                psf_sub["diff_psf_mag"],
                yerr=psf_sub["diff_psf_mag_err"],
                fmt="o-",
                markersize=4.0,
                linewidth=1.2,
                capsize=2.5,
                elinewidth=0.9,
                color=color,
                label=f"{filt}-band",
                alpha=0.95,
            )

        if not forced_sub.empty:
            ax.errorbar(
                forced_sub["MJD"],
                forced_sub["diff_psf_mag"],
                yerr=forced_sub["diff_psf_mag_err"],
                fmt="D",
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linestyle="none",
                capsize=2.5,
                elinewidth=0.9,
                color=color,
                alpha=0.95,
            )

    if ref_df is not None:
        for ref_filter in sorted(ref_df["filter"].astype(str).unique()):
            ref_sub = ref_df[ref_df["filter"].astype(str) == ref_filter].sort_values("mjd")
            ref_color = color_map.get(canonical_filter_name(ref_filter), "black")
            ax.errorbar(
                ref_sub["mjd"],
                ref_sub["mag"],
                yerr=ref_sub["magerr"],
                fmt="s",
                markersize=3.8,
                markerfacecolor="none",
                markeredgewidth=0.9,
                linestyle="--",
                linewidth=1.0,
                capsize=2.0,
                elinewidth=0.8,
                color=ref_color,
                alpha=0.8,
                label=f"Ref {ref_filter}",
            )

    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("AB Magnitude (diff_psf_mag)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linestyle="-")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(title="Filter", frameon=True, loc="best")
    ax.plot(
        [],
        [],
        "D",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="none",
        label="Forced_PSF",
    )
    if target_canonical_filter:
        ax.set_title(f"{target_canonical_filter}-band light curve")
    ax.legend(title="Filter / Type", frameon=True, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-filter light curves from photometry CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ep260321a_photometry.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/ep260321a_lightcurve.png"),
        help="Output figure path (.pdf or .png).",
    )
    parser.add_argument(
        "--diff-psf-type",
        type=str,
        default="",
        help="Optional filter on diff_PSF_type column. Use empty string to disable.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window.",
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=None,
        help=(
            "Optional reference lightcurve CSV path. "
            "Expected columns: mjd, mag, magerr, filter."
        ),
    )
    parser.add_argument(
        "--separate-filters",
        nargs="+",
        default=None,
        help=(
            "Create separate plots for canonical bands (e.g., i g r). "
            "Each plot includes matching reference filters."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_type = args.diff_psf_type if args.diff_psf_type else None
    if args.separate_filters:
        for band in args.separate_filters:
            canonical_band = canonical_filter_name(band)
            band_output = args.output.with_name(
                f"{args.output.stem}_{canonical_band}{args.output.suffix}"
            )
            plot_lightcurve(
                csv_path=args.input,
                output_path=band_output,
                show=args.show,
                diff_psf_type=selected_type,
                reference_csv_path=args.reference_input,
                target_canonical_filter=canonical_band,
            )
    else:
        plot_lightcurve(
            csv_path=args.input,
            output_path=args.output,
            show=args.show,
            diff_psf_type=selected_type,
            reference_csv_path=args.reference_input,
        )
