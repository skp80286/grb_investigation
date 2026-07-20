#!/usr/bin/env python3
"""Read r-band data from a CSV and fit a power law to AB magnitude."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a power law to r-band AB magnitudes from a CSV file."
    )
    parser.add_argument(
        "--csv",
        default="data/GRB260516D_AB.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--band",
        default="r",
        help="Filter name to select from the Filt column (case-insensitive).",
    )
    parser.add_argument(
        "--output",
        default="plots/r_band_powerlaw_fit.png",
        help="Path to the output plot file.",
    )
    return parser.parse_args()


def load_r_band_data(csv_path, band):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "Filt" not in df.columns:
        raise ValueError("CSV file must contain a 'Filt' column.")
    if "Times" not in df.columns:
        raise ValueError("CSV file must contain a 'Times' column.")
    if "AB" not in df.columns:
        raise ValueError("CSV file must contain an 'AB' column.")

    band_mask = df["Filt"].astype(str).str.strip().str.lower() == band.lower()
    df_r = df.loc[band_mask].copy()
    if df_r.empty:
        raise ValueError(f"No rows found for filter '{band}' in {csv_path}.")

    df_r["Times"] = pd.to_numeric(df_r["Times"], errors="coerce")
    df_r["AB"] = pd.to_numeric(df_r["AB"], errors="coerce")
    if "AB_Err" in df_r.columns:
        df_r["AB_Err"] = pd.to_numeric(df_r["AB_Err"], errors="coerce")
    else:
        df_r["AB_Err"] = np.nan

    df_r = df_r.dropna(subset=["Times", "AB"])
    if df_r.empty:
        raise ValueError(f"No valid numeric r-band data found in {csv_path}.")

    return df_r.sort_values(by="Times")


def powerlaw_mag(t, m0, alpha, t_ref=1.0):
    return m0 + alpha * np.log10(t / t_ref)


def fit_powerlaw(t, m, m_err):
    t_ref = np.median(t)
    p0 = [m.mean(), 1.0]
    sigma = m_err if np.any(np.isfinite(m_err)) else None
    popt, pcov = curve_fit(
        lambda t, m0, alpha: powerlaw_mag(t, m0, alpha, t_ref),
        t,
        m,
        sigma=sigma,
        absolute_sigma=sigma is not None,
        p0=p0,
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, t_ref


def plot_fit(df_r, popt, t_ref, output_path, band):
    t = df_r["Times"].to_numpy()
    m = df_r["AB"].to_numpy()
    m_err = df_r["AB_Err"].to_numpy()

    t_fit = np.logspace(np.log10(t.min()), np.log10(t.max()), 200)
    m_fit = powerlaw_mag(t_fit, *popt, t_ref=t_ref)

    try:
        plt.style.use(["science", "high-vis"])
    except (OSError, ImportError, ValueError):
        plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(2.0)
    fig.patch.set_facecolor("white")

    ax.errorbar(
        t,
        m,
        yerr=np.where(np.isfinite(m_err), m_err, None),
        fmt="o",
        color="tab:red",
        mec="black",
        capsize=3,
        label=f"{band} band data",
    )
    ax.plot(t_fit, m_fit, color="tab:blue", lw=2, label="Power-law fit")

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AB magnitude")
    ax.set_title(f"{band} band AB magnitude power-law fit")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor("black")

    ax.tick_params(direction="in", length=6, width=1.5, colors="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="black")
    plt.close(fig)


def main():
    args = parse_args()
    df_r = load_r_band_data(args.csv, args.band)

    t = df_r["Times"].to_numpy(dtype=float)
    mag = df_r["AB"].to_numpy(dtype=float)
    mag_err = df_r["AB_Err"].to_numpy(dtype=float)
    if not np.any(np.isfinite(mag_err)):
        mag_err = np.ones_like(mag) * np.std(mag - np.mean(mag))

    popt, perr, t_ref = fit_powerlaw(t, mag, mag_err)
    m0, alpha = popt
    dm0, dalpha = perr

    print("Power-law fit to r-band AB magnitude:")
    print(f"  m(t) = m0 + alpha * log10(t / t_ref)")
    print(f"  t_ref = {t_ref:.3g} s")
    print(f"  m0 = {m0:.4f} +/- {dm0:.4f}")
    print(f"  alpha = -{alpha/2.5:.4f} +/- {dalpha/2.5:.4f}")

    output_dir = os.path.dirname(args.output) or "plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_fit(df_r, popt, t_ref, args.output, args.band)
    print(f"Saved fit plot to {args.output}")


if __name__ == "__main__":
    main()
