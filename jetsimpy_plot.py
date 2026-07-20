# jetsimpy_plot.py — Plot GRB afterglow model light curves and overlay observations
#
# Description:
# - Reads observed photometry from a CSV (`--obsfile`) with columns: Filt, Times, Fluxes, FluxErrs.
# - Builds a jetsimpy model from `--params` (dict/JSON), supports log-prefixed keys (e.g., loge0 → e0).
# - Computes multi-band model fluxes, overlays observations, and saves plots and a log in `<obsdir>/output/`.
#
# CLI:
#   --obsfile  Path to observations CSV.
#   --params   JSON/dict of model parameters; include `jetType` (gaussian|powerlaw|tophat) and `z`.
#   --spectrum Plot F_nu vs nu at 1, 10, …, 1e6 s using Jet.Flux() (no light-curve plot).
#   --label    Optional label (reserved).
#
# Outputs:
#   <obsdir>/output/lc_afterflow_obs_matching.pdf and .png
#   <obsdir>/output/jetsimpy_plot_.log
#
# Example:
#   python jetsimpy_plot.py --obsfile data/GRB250916A_cons.csv  \
# --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, \
# n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'
#
# You can also use this code as a library.
# Example:
# import jetsimpy_plot as jsim
# %matplotlib inline # if you want to show plots interactively in a jupyter notebook
# import matplotlib.pyplot as plt

# params={'jetType': 'tophat', 'e0': 4.87e52, 'epsb': 0.0448, 'epse': 0.3981, 'n0': 0.0032, 'thc': 0.0623, 'thv': 0.0014, 'p': 2.3578, 'loglf': 100, 'A': 0, 's': 0, 'z': 2.011}
# jsim.lc_plot(basedir="output", params=params, observed_data='data/GRB250916A_cons.csv', show_plot=True, save_plot=False)

import copy
import json
import os
import random
import datetime
import warnings

import sys

import jetsimpy
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
from scipy import stats
from scipy.optimize import curve_fit, minimize, newton
import logging
import argparse
from jsonargparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from lc_plot_settings import lc_plot_settings_default

######################

logger = logging.getLogger(__name__)

# jetsimpy: Flux is erg/s/cm^2 integrated over ν; FluxDensity divides by this for mJy
_MJY_PER_CGS_FNU = 1e-26
_SEC_PER_DAY = 86400.0

# Serif stack for publication-style figures (PDF embedding + math consistency)
_PAPER_SERIF_RC = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
        "serif",
    ],
    "mathtext.fontset": "dejavuserif",
}


def _normalize_ul_column(df):
    """
    Ensure 'UL' column exists and blank/NaN values are treated as 'N'.
    Creates 'UL' column with all 'N' values if it doesn't exist.
    Fills blank and NaN values with 'N'.
    Mutates df in place and returns it.
    """
    if "UL" not in df.columns:
        df["UL"] = "N"
    else:
        # Replace NaN and empty strings with "N"
        df["UL"] = df["UL"].fillna("N")
        df["UL"] = df["UL"].astype(str).str.strip()
        df.loc[df["UL"] == "", "UL"] = "N"
    return df


def _expand_jetsimpy_params_inplace(params):
    """
    Apply log* and expthc/expthv conversions. Mutates ``params`` in place.
    """
    for k in list(params.keys()):
        if isinstance(k, str) and k.startswith("log"):
            new_k = k[3:]
            params[new_k] = 10 ** params[k]
            params.pop(k)

    if "expthc" in params:
        params["thc"] = np.log10(params["expthc"])
        params.pop("expthc")
    if "expthv" in params:
        params["thv"] = np.log10(params["expthv"])
        params.pop("expthv")


def _jet_and_P(params):
    """
    Build jetsimpy.Jet and emissivity parameter dict P from *physical* params.

    Mutates ``params`` in place (log-prefixed keys, expthc/expthv). Pass a copy
    if the caller needs the original dict unchanged.
    """
    _expand_jetsimpy_params_inplace(params)
    dl = cosmo.luminosity_distance(params["z"]).to("Mpc").value

    P = dict(
        eps_e=params["epse"],
        eps_b=params["epsb"],
        p=params["p"],
        theta_v=params["thv"],
        d=dl,
        z=params["z"],
    )

    jet_P = dict(
        Eiso=params["e0"],
        lf=params["lf"],
        theta_c=params["thc"],
        n0=params["n0"],
        A=params["A"],
        s=params["s"],
    )

    if params["jetType"] == "gaussian":
        jetProfile = jetsimpy.Gaussian(jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"])
    elif params["jetType"] == "powerlaw":
        jetProfile = jetsimpy.PowerLaw(
            jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"], s=jet_P["s"]
        )
    else:
        jetProfile = jetsimpy.TopHat(jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"])

    jet = jetsimpy.Jet(
        jetProfile,
        nwind=jet_P["A"],
        nism=jet_P["n0"],
        grid=jetsimpy.ForwardJetRes(jet_P["theta_c"], 129),
        spread=True,
        tmin=1.0,
        tmax=3.2e9,
        tail=True,
        cal_level=1,
        rtol=1e-6,
        cfl=0.9,
    )
    return jet, P


def model(obs_time, obs_nu, params):
    p = copy.deepcopy(params)
    jet, P = _jet_and_P(p)
    model_flux = jet.FluxDensity(
        obs_time,
        obs_nu,
        P,
        model="sync",
        rtol=1e-3,
        max_iter=100,
        force_return=True,
    )
    return model_flux


# multipliers = {'X-ray(1keV)': 10.0, 'X-ray(10keV)': 100.0, 'g': 1.0, 'L': 1, 'R': 1,'r': 1,
#'i': 8.0, 'u': 8.0, 'z': 16.0, 'J': 32.0,
#'radio(1.3GHz)': 100.0, 'radio(6GHz)': 400, 'radio(10GHz)': 1500, 'radio(15GHz)': 2000}
multipliers = {
    "X-ray(10keV)": 32,
    "u": 1,
    "g": 2,
    "VT_B": 4,
    "r": 8,
    "R": 16,
    "i": 32,
    "z'": 32,
    "VT_R": 64,
    "J": 128,
    # "6GHz": 64,
    # "radio(1.3GHz)": 10,
    # "radio(3GHz)": 25,
    # "radio(6GHz)": 50,
    # "radio(10GHz)": 100,
    "radio(15.5GHz)": 1024,
    # "radio(75GHz)": 300,
    # "radio(90GHz)": 500,
}

filt_freqs = {
    # "i'": 3.843e14,
    "i": 3.98913e14,
    # "z'": 3.225e14,
    "z": 3.46e14,
    "VT_B": 5.45077e14,
    # "VT_R": 3.63385e14,
    # "r'": 4.732e14,
    "r": 4.8384e14,
    "J": 2.40161e14,
    # "g'": 6.087e14,
    "g": 6.249e14,
    # "R": 4.67914e14,
    "L": 5.55516e14,
    "u": 8.1178e14,
    # "SAO-R": 4.556231e13,
    "X-ray(10keV)": 2.41799e18,
    # "X-ray(1keV)": 2.41799e17,
    "radio(1.3GHz)": 1.3e9,
    "radio(3GHz)": 3e9,
    "radio(6GHz)": 6e9,
    # "6GHz": 6e9,
    "radio(10GHz)": 1e10,
    "radio(15GHz)": 1.5e10,
    "radio(15.5GHz)": 1.55e10,
    "radio(75GHz)": 7.5e10,
    "radio(90GHz)": 9e10,
}

# cmap = matplotlib.colormaps.get_cmap('rainbow_r')  # or 'plasma', 'cividis', 'magma'
# colors = cmap(np.linspace(0, 1, len(filt_freqs)))

# colors=['tab:purple', 'darkgreen', 'tab:red', 'darkgoldenrod', 'olive', 'royalblue', '#580F41', 'orange', 'cyan']
band_colors = {
    "X-ray(10keV)": "darkviolet",
    "u": "teal",
    "u'": "teal",
    "VT_B": "royalblue",
    "g": "darkgreen",
    "g'": "darkgreen",
    "r": "tab:red",
    "r'": "tab:red",
    "i": "darkgoldenrod",
    "i'": "darkgoldenrod",
    "z": "peru",
    "z'": "peru",
    "VT_R": "orange",
    "R": "magenta",
    "J": "olive",
    "radio(1.3GHz)": "darkgreen",
    "radio(3GHz)": "cornflowerblue",
    "radio(6GHz)": "peru",
    "radio(10GHz)": "#562778",
    "radio(15GHz)": "#441E5F",
    "radio(15.5GHz)": "deepskyblue",
    "radio(75GHz)": "olive",
    "radio(90GHz)": "#9141CA",
}

band_secondary_colors = {
    "X-ray(10keV)": "lavender",
    "u": "mediumturquoise",
    "VT_B": "lightsteelblue",
    "g": "mediumaquamarine",
    "g'": "mediumaquamarine",
    "r": "lightcoral",
    "r'": "lightcoral",
    "i": "khaki",
    "i'": "khaki",
    "z": "peachpuff",
    "z'": "peachpuff",
    "VT_R": "peachpuff",
    "R": "plum",
    "J": "olive",
    "radio(1.3GHz)": "mediumaquamarine",
    "radio(3GHz)": "cornflowerblue",
    "radio(6GHz)": "peachpuff",
    "radio(10GHz)": "#5F3E77",
    "radio(15GHz)": "#4D385C",
    "radio(15.5GHz)": "lightblue",
    "radio(75GHz)": "tan",
    "radio(90GHz)": "#A97BCA",
}

"""
    band_colors = {
        "radio(15.5GHz)": "#4B0082",  # deep purple
        "L": "#5A0000",               # very dark red (near-IR, long λ)
        "J": "#8B0000",               # dark red / near-IR
        "z": "#A00000",               # deep maroon
        "R": "#E41A1C",               # red
        "VT_R": "#FF7F00",            # orange-red
        "i": "#D95F02",               # amber
        "r": "#4DAF4A",               # green
        "g": "#00A6D6",               # cyan
        "VT_B": "#377EB8",            # blue
        "u": "#984EA3",               # violet / near-UV
        "X-ray(10keV)": "#000000"     # black (extreme high-energy)
    }
"""


def lc_plot(
    basedir,
    median_params,
    sig3_params,
    observed_data,
    show_plot=False,
    save_plot=True,
    hide_z_text=False,
    plot_settings=None,
):
    """Plot a modeled light curve using ``plot_settings`` or the default."""
    settings = lc_plot_settings_default if plot_settings is None else plot_settings
    required_settings = {"xlim", "ylim", "multipliers", "filt_freqs", "band_colors", "band_secondary_colors"}
    missing_settings = required_settings.difference(settings)
    if missing_settings:
        raise ValueError("plot_settings is missing required keys: " + ", ".join(sorted(missing_settings)))
    xlim = settings["xlim"]
    ylim = settings["ylim"]
    multipliers = settings["multipliers"]
    filt_freqs = settings["filt_freqs"]
    band_colors = settings["band_colors"]
    band_secondary_colors = settings["band_secondary_colors"]
    plt.style.use(["science", "high-vis"])

    mpl.rcParams.update(
        {
            **_PAPER_SERIF_RC,
            "font.size": 5,  # minimum allowed by Nature
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,  # embed fonts as TrueType
            "ps.fonttype": 42,
            # "figure.dpi": 300,  # ensure high-res bitmap export when needed
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
        }
    )

    # Time and Frequencies
    ta = 1.0e3
    tb = 1.0e7
    t = np.geomspace(ta, tb, num=100)

    df_allobs = pd.read_csv(observed_data)
    # Convert numeric columns to numeric types, handling invalid values
    df_allobs["Times"] = pd.to_numeric(df_allobs["Times"], errors="coerce")
    df_allobs["Fluxes"] = pd.to_numeric(df_allobs["Fluxes"], errors="coerce")
    df_allobs["FluxErrs"] = pd.to_numeric(df_allobs["FluxErrs"], errors="coerce")
    # Normalize UL column: treat missing/blank as "N"
    df_allobs = _normalize_ul_column(df_allobs)
    available_bands = (
        set(df_allobs["Filt"].dropna()) if "Filt" in df_allobs.columns else set()
    )
    logger.info(
        f"lc_plot: len(median_params)={len(median_params)}, len(sig3_parmas)={len(sig3_params)}, len(df_allobs)={len(df_allobs)}"
    )

    # Precompute model fluxes for each band and for median + sig3 samples in parallel
    bands_to_compute = [
        (band, nu)
        for band, nu in sorted(filt_freqs.items(), key=lambda x: -x[1])
        if band in multipliers
    ]

    precomputed = {}  # keys: (band, 'median') or (band, idx)
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for band, nu in bands_to_compute:
            future = executor.submit(model, t, [nu], median_params)
            future_map[future] = (band, "median")
            for i, params in enumerate(sig3_params):
                future = executor.submit(model, t, [nu], params)
                future_map[future] = (band, i)

        for fut in as_completed(future_map):
            band, tag = future_map[fut]
            try:
                res = np.array(fut.result())
            except Exception as e:
                logger.error(f"model failed during precompute for band {band}: {e}")
                res = None
            precomputed[(band, tag)] = res

    # Print fluxes at selected observer times in days for each band.
    flux_times_days = [1, 2, 4, 8, 16]
    flux_times_seconds = [d * 86400.0 for d in flux_times_days]
    logger.info("band,frequency_hz,time_days,time_seconds,flux_mjy")
    for band, nu in sorted(filt_freqs.items(), key=lambda x: -x[1]):
        if band not in multipliers:
            continue
        try:
            flux_values = model(flux_times_seconds, [nu], median_params)
            flux_values = np.array(flux_values).flatten()
        except Exception as e:
            logger.error(f"model failed for band {band} at selected times: {e}")
            continue
        for day, sec, flux in zip(flux_times_days, flux_times_seconds, flux_values):
            logger.info(f"{band},{nu},{day},{sec},{flux}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot the model curves - expected lightcurve from jetsimpy
    j = -1
    for i, (band, nu) in enumerate(sorted(filt_freqs.items(), key=lambda x: -x[1])):
        if band in multipliers:
            multiplier = multipliers[band]
        else:
            continue
        j += 1

        logger.info(f"Calculating for frequency: {nu}")
        Fnu_model = []
        # Retrieve precomputed median result
        try:
            Fnu_model = precomputed.get((band, "median"))
            if Fnu_model is None:
                raise RuntimeError("No precomputed median model for band")

            # plot sig3 samples (if any)
            for idx in range(len(sig3_params)):
                Fnu_sig3 = precomputed.get((band, idx))
                if Fnu_sig3 is None:
                    logger.debug(f"Missing precomputed sig3 for band={band}, idx={idx}")
                    continue
                ax.plot(
                    t,
                    Fnu_sig3 * multiplier,
                    linewidth=1.0,
                    linestyle="-",
                    color=band_secondary_colors.get(band, "#C5C6C7"),
                    alpha=0.2,
                )

            ax.plot(
                t,
                Fnu_model * multiplier,
                linewidth=1.0,
                linestyle="-",
                label=f"{band} x {multiplier}",
                color=band_colors.get(band, "#616569"),
                alpha=1,
            )
        except Exception as e:
            logger.error(f"model failed for band {band}; {e}")
            return -1e100

    # plot the actual observations
    j = -1
    for i, (band, nu) in enumerate(sorted(filt_freqs.items(), key=lambda x: -x[1])):
        if band in multipliers:
            multiplier = multipliers[band]
        else:
            continue
        if band not in available_bands:
            logger.info(f"Skipping band={band}; not present in df_allobs['Filt'].")
            continue
        j += 1

        Fnu_allobs = (
            df_allobs[(df_allobs["Filt"] == band) & (df_allobs["UL"] == "N")][
                ["Times", "Fluxes", "FluxErrs"]
            ]
            .sort_values(by="Times")
            .to_numpy()
        )
        if len(Fnu_allobs) > 0:
            # logger.info(f"Skipping detections for band={band}; no UL='N' rows.")
            # continue
            logger.info(
                f"Plotting band={band}, {len(Fnu_allobs)} rows, err={Fnu_allobs[:, 2]}."
            )

            ax.errorbar(
                Fnu_allobs[:, 0],
                Fnu_allobs[:, 1] * multiplier,
                yerr=Fnu_allobs[:, 2] * multiplier,
                fmt="o",
                markersize=4,
                alpha=1,
                color=band_colors.get(band, "#616569"),
                mec="black",
                elinewidth=0.5,
                capsize=2,
            )

        Fnu_ul_obs = (
            df_allobs[(df_allobs["Filt"] == band) & (df_allobs["UL"] == "Y")][
                ["Times", "Fluxes", "FluxErrs"]
            ]
            .sort_values(by="Times")
            .to_numpy()
        )
        if len(Fnu_ul_obs) > 0:
            logger.info(f"Plotting upper limit band={band}, {len(Fnu_ul_obs)} rows.")

            """
            ax.scatter(
                    Fnu_ul_obs[:,0], Fnu_ul_obs[:,1]*multiplier,
                    marker='v',
                    s=4, alpha=1,
                    c=colors[j % len(colors)], 
                    edgecolors='black', 
            )
            """

            # Plot upper limits with arrows pointing down
            plt.errorbar(
                Fnu_ul_obs[:, 0],
                Fnu_ul_obs[:, 1] * multiplier,
                yerr=None,
                fmt="v",
                markersize=6,
                alpha=1,
                color=band_colors.get(band, "#616569"),
                mec="black",
                elinewidth=0.5,
                capsize=2,
                uplims=True,
            )

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="y", which="minor", length=3, width=0.5)
    ax.tick_params(axis="x", which="minor", length=3, width=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$F_\nu$ (mJy)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Create text content with all Z dictionary values
    z_text = ""
    for key, value in median_params.items():
        if key in ["specType", "z", "E0"]:
            continue
            # Skip function objects, just show the key
            # z_text += f"{key}: {type(value).__name__}\n"
        elif (key == "s" and value == 0) or (key == "logA" and value == 0):
            continue
        else:
            z_text += "\n"
            # Format numerical values
            if key.startswith("log"):
                key = key[3:]
                value = 10**value
            if isinstance(value, (int, float)):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    z_text += f"{key}: {value:.2e}"
                else:
                    z_text += f"{key}: {value:.4f}"
            else:
                z_text += f"{key}: {value}"

    if not hide_z_text:
        # Add textbox with all Z dictionary values
        ax.text(
            0.98,
            0.02,
            z_text,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.5, edgecolor="none"
            ),
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=12,
        )

    # marker_text = "*  Observations used for fitting\nx  All observations\nDashed lines show the best fit"
    # ax.text(0.2, 0.02, marker_text, transform=ax.transAxes,
    #        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black"),
    #        verticalalignment='bottom', fontsize=10, fontfamily='monospace')

    ax.legend(edgecolor="none", loc="lower left", ncol=2)
    fig.tight_layout()

    if save_plot:
        logging.info(
            f"Saving lightcurve fit plot to: {basedir}/lc_afterflow_obs_matching.pdf"
        )
        fig.savefig(
            f"{basedir}/lc_afterflow_obs_matching.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        logging.info(
            f"Saving lightcurve fit plot to: {basedir}/lc_afterflow_obs_matching.png"
        )
        fig.savefig(
            f"{basedir}/lc_afterflow_obs_matching.png",
            format="png",
            bbox_inches="tight",
            dpi=300,
        )
    if show_plot:
        plt.show()
    plt.close(fig)


# Observer times (s) used when overlaying spectrum observations from a photometry CSV.
SPECTRUM_PLOT_TIME_EPOCHS = np.array(
    [
        1e3,
        1e4,
        # 31500.0,
        # 32600.0,
        # 38500.0,
        35244.0,
        # 42300.0,
        # 82900.0,
        # 118500.0,
        189500.0,
        1e6,
    ],
    dtype=float,
)


def build_spectrum_epoch_observations(df_allobs, epochs, dt_sec=500.0):
    """
    For each entry in ``epochs``, collect detections with ``Times`` within ±``dt_sec``
    seconds of that epoch.

    Each returned observation is ``(nu_hz, flux_mjy, flux_err_mjy)`` from columns
    ``Freqs``, ``Fluxes``, ``FluxErrs``. If ``UL`` is present, only rows with
    ``UL == 'N'`` are used.

    Returns a list of length ``len(epochs)``; entries are lists (possibly empty).
    """
    df = df_allobs.copy()
    for col in ("Times", "Freqs", "Fluxes", "FluxErrs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize UL column: treat missing/blank as "N"
    df = _normalize_ul_column(df)
    df = df[df["UL"].astype(str) == "N"]
    if "Freqs" not in df.columns:
        return [[] for _ in np.asarray(epochs, dtype=float)]

    t_arr = df["Times"].to_numpy(dtype=float)
    out = []
    for t0 in np.asarray(epochs, dtype=float):
        m = np.abs(t_arr - t0) <= float(dt_sec)
        sub = df.loc[m]
        if len(sub) == 0:
            out.append([])
            continue
        pts = []
        for _, row in sub.iterrows():
            nu = row["Freqs"]
            fl = row["Fluxes"]
            fe = row["FluxErrs"]
            if not (np.isfinite(nu) and np.isfinite(fl) and np.isfinite(fe)):
                continue
            pts.append((float(nu), float(fl), float(fe)))
        out.append(pts)
    return out


def spectrum_plot(
    basedir,
    median_params,
    show_plot=False,
    save_plot=True,
    n_freq_bins=100,
    time_epochs=None,
    epoch_observations=None,
):
    """
    Plot modeled afterglow spectrum F_ν vs ν at several epochs using ``Jet.Flux``.

    Frequency grid: logarithmic edges from 1e9 to 1e19 Hz. For each bin
    ``[ν_lo, ν_hi]``, integrated flux ``Flux(t, ν_lo, ν_hi)`` (erg/s/cm²) is
    divided by ``(ν_hi - ν_lo)`` and by ``1e-26`` to match mJy, consistent with
    ``FluxDensity`` for narrow bins.

    Time epochs: if ``time_epochs`` is omitted, uses 1 s, 10 s, …, 1e6 s (seven
    log-spaced points). Optional ``epoch_observations`` is a list of the same
    length as ``time_epochs``; each element is a list of
    ``(nu_hz, flux_mjy, flux_err_mjy)`` tuples plotted at that ν with the same
    color as the model curve for that epoch. Legend labels show observer time in
    days.
    """
    plt.style.use(["science", "high-vis"])

    mpl.rcParams.update(
        {
            **_PAPER_SERIF_RC,
            "font.size": 5,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
        }
    )

    p = copy.deepcopy(median_params)
    jet, P = _jet_and_P(p)

    nu_edges = np.geomspace(1e9, 1e19, n_freq_bins + 1)
    nu_lo = nu_edges[:-1]
    nu_hi = nu_edges[1:]
    dnu = nu_hi - nu_lo
    nu_c = np.sqrt(nu_lo * nu_hi)

    # 1 s, 10 s, …, 1e6 s (seven decades) when custom epochs are not provided
    times = (
        np.asarray(time_epochs, dtype=float)
        if time_epochs is not None
        else np.geomspace(1.0, 1.0e6, num=7)
    )
    n_ep = len(times)
    if epoch_observations is None:
        obs_by_epoch = [[] for _ in range(n_ep)]
    else:
        if len(epoch_observations) != n_ep:
            raise ValueError(
                f"epoch_observations must have length {n_ep} (same as time epochs), "
                f"got {len(epoch_observations)}"
            )
        obs_by_epoch = list(epoch_observations)

    colors = mpl.cm.viridis(np.linspace(0.15, 0.95, n_ep))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for it, t_obs in enumerate(times):
        fnu = np.empty_like(nu_c, dtype=float)
        for i in range(len(nu_c)):
            fband = jet.Flux(
                t_obs,
                float(nu_lo[i]),
                float(nu_hi[i]),
                P,
                model="sync",
                rtol=1e-3,
                max_iter=100,
                force_return=True,
            )
            fnu[i] = fband / dnu[i] / _MJY_PER_CGS_FNU

        ax.plot(
            nu_c,
            fnu,
            color=colors[it],
            linestyle="-",
            label=f"{t_obs / _SEC_PER_DAY:.4g} d",
        )

        obs_list = obs_by_epoch[it]
        if obs_list:
            nu_obs = np.array([o[0] for o in obs_list], dtype=float)
            f_obs = np.array([o[1] for o in obs_list], dtype=float)
            err_obs = np.array([o[2] for o in obs_list], dtype=float)
            good = np.isfinite(nu_obs) & np.isfinite(f_obs) & np.isfinite(err_obs)
            nu_obs, f_obs, err_obs = nu_obs[good], f_obs[good], err_obs[good]
            if len(nu_obs) > 0:
                ax.errorbar(
                    nu_obs,
                    f_obs,
                    yerr=err_obs,
                    fmt="o",
                    color=colors[it],
                    ecolor=colors[it],
                    elinewidth=0.5,
                    capsize=2,
                    markersize=3.5,
                    zorder=5,
                )

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (Hz)")
    ax.set_ylabel(r"$F_\nu$ (mJy)")
    ax.set_title(r"Afterglow spectrum (median jet parameters)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(edgecolor="none", loc="best", ncol=2)
    fig.tight_layout()

    if save_plot:
        pdf_path = f"{basedir}/spectrum_afterglow_epochs.pdf"
        png_path = f"{basedir}/spectrum_afterglow_epochs.png"
        logging.info("Saving spectrum plot to: %s", pdf_path)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        logging.info("Saving spectrum plot to: %s", png_path)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


def _estimate_break_frequencies(nu, fnu):
    """
    Estimate nu_m and nu_c from a sampled synchrotron spectrum.

    - nu_m: frequency at peak F_nu
    - nu_c: post-peak break where local slope steepens by ~0.5
    """
    nu = np.asarray(nu, dtype=float)
    fnu = np.asarray(fnu, dtype=float)
    valid = np.isfinite(nu) & np.isfinite(fnu) & (nu > 0) & (fnu > 0)
    if np.count_nonzero(valid) < 8:
        return np.nan, np.nan

    nu = nu[valid]
    fnu = fnu[valid]
    lognu = np.log10(nu)
    logf = np.log10(fnu)

    i_m = int(np.argmax(logf))
    nu_m = nu[i_m]

    # Need post-peak points to infer cooling break.
    if i_m >= len(nu) - 5:
        return nu_m, np.nan

    alpha = np.gradient(logf, lognu)  # local spectral slope dlogF/dlognu
    lo = min(i_m + 1, len(alpha) - 2)
    hi = min(i_m + 4, len(alpha))
    alpha_post_peak = np.median(alpha[lo:hi]) if hi > lo else alpha[lo]
    target_alpha = alpha_post_peak - 0.5

    cand = np.arange(i_m + 2, len(alpha) - 1)
    if len(cand) == 0:
        return nu_m, np.nan

    steep = cand[alpha[cand] < alpha_post_peak - 0.2]
    search = steep if len(steep) > 0 else cand
    i_c = search[np.argmin(np.abs(alpha[search] - target_alpha))]
    nu_c = nu[int(i_c)]
    return nu_m, nu_c


def compute_break_frequencies_tophat_analytical(
    z, p, eps_e, eps_b, e0_erg, n0, t_seconds
):
    """
    Analytical ISM forward-shock synchrotron break frequencies vs observer time (tophat scalings).

    Uses the same scaling-law forms as sync_freq_evolution in GRB250704B_170817_jetsimpy.ipynb:
    observer time in days td = t/(86400 s), isotropic equivalent energy E52 = E_iso / 10^52 erg.

    Returns arrays shaped like ``t_seconds``.
    """
    td = np.asarray(t_seconds, dtype=float) / 86400.0
    E52 = e0_erg / 1e52
    nu_m = (
        5.1e15
        * (1 + z) ** 0.5
        * ((p - 2) / (p - 1)) ** 2
        * eps_e**2
        * eps_b**0.5
        * E52**0.5
        * td ** (-1.5)
    )
    nu_c = (
        2.7e12
        * (1 + z) ** (-0.5)
        * eps_b ** (-1.5)
        * E52 ** (-0.5)
        * n0 ** (-1)
        * td ** (-0.5)
    )
    return nu_m, nu_c


def compute_break_frequencies_from_spectrum(jet, P, times, n_freq_bins=160):
    """
    Infer ν_m and ν_c at each observer time from synchrotron spectra built with ``jet.Flux``.

    Frequency grid: 1e9–1e19 Hz in ``n_freq_bins`` logarithmic bins; bin flux is converted to
    F_ν by dividing by bin width. Breaks are estimated with ``_estimate_break_frequencies``.

    Parameters
    ----------
    jet, P
        From ``_jet_and_P(median_params)``.
    times : array_like
        Observer times (s).

    Returns
    -------
    nu_m_all, nu_c_all : ndarray
        Same length as ``times``.
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    nu_edges = np.geomspace(1e9, 1e19, n_freq_bins + 1)
    nu_lo = nu_edges[:-1]
    nu_hi = nu_edges[1:]
    dnu = nu_hi - nu_lo
    nu_cen = np.sqrt(nu_lo * nu_hi)

    nu_m_all = np.full(n_time, np.nan, dtype=float)
    nu_c_all = np.full(n_time, np.nan, dtype=float)

    for it, t_obs in enumerate(times):
        fnu = np.empty_like(nu_cen, dtype=float)
        for i in range(len(nu_cen)):
            fband = jet.Flux(
                t_obs,
                float(nu_lo[i]),
                float(nu_hi[i]),
                P,
                model="sync",
                rtol=1e-3,
                max_iter=100,
                force_return=True,
            )
            fnu[i] = fband / dnu[i] / _MJY_PER_CGS_FNU

        nu_m, nu_c = _estimate_break_frequencies(nu_cen, fnu)
        nu_m_all[it] = nu_m
        nu_c_all[it] = nu_c

    return nu_m_all, nu_c_all


def compute_break_frequencies_timeseries(median_params, times, n_freq_bins=160):
    """
    Break-frequency time series: analytical ISM scalings if ``jetType`` is tophat (case-insensitive),
    otherwise inferred from ``Jet.Flux`` synchrotron spectra (other jet profiles).
    """
    probe = copy.deepcopy(median_params)
    _expand_jetsimpy_params_inplace(probe)
    if str(probe["jetType"]).lower() == "tophat":
        return compute_break_frequencies_tophat_analytical(
            probe["z"],
            probe["p"],
            probe["epse"],
            probe["epsb"],
            probe["e0"],
            probe["n0"],
            times,
        )
    pc = copy.deepcopy(median_params)
    jet, P = _jet_and_P(pc)
    return compute_break_frequencies_from_spectrum(jet, P, times, n_freq_bins)


def _figure_break_frequency_evolution(times, nu_m_all, nu_c_all, title):
    """Shared figure: ν_m and ν_c vs observer time (log–log). Caller sets matplotlib style."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mask_m = np.isfinite(nu_m_all) & (nu_m_all > 0)
    mask_c = np.isfinite(nu_c_all) & (nu_c_all > 0)
    if np.any(mask_m):
        ax.plot(times[mask_m], nu_m_all[mask_m], color="tab:blue", label=r"$\nu_m$")
    if np.any(mask_c):
        ax.plot(times[mask_c], nu_c_all[mask_c], color="tab:red", label=r"$\nu_c$")

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"Break frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(edgecolor="none", loc="best")
    fig.tight_layout()
    return fig


def break_frequency_evolution_plot(
    basedir,
    median_params,
    show_plot=False,
    save_plot=True,
    n_freq_bins=160,
    n_time=60,
):
    """
    Plot synchrotron break-frequency evolution (ν_m, ν_c) vs time.

    For ``jetType=='tophat'``, ν_m and ν_c use analytical scaling laws
    (``compute_break_frequencies_tophat_analytical``). For other jet types, values are inferred
    from modeled synchrotron spectra (``compute_break_frequencies_from_spectrum``).
    """
    plt.style.use(["science", "high-vis"])

    mpl.rcParams.update(
        {
            **_PAPER_SERIF_RC,
            "font.size": 5,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.9,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
        }
    )

    times = np.geomspace(1.0, 1.0e6, num=n_time)
    nu_m_all, nu_c_all = compute_break_frequencies_timeseries(
        median_params, times, n_freq_bins=n_freq_bins
    )

    p = copy.deepcopy(median_params)
    _expand_jetsimpy_params_inplace(p)
    if str(p["jetType"]).lower() == "tophat":
        title = (
            r"Evolution of $\nu_m$ and $\nu_c$ (analytical scalings, median parameters)"
        )
    else:
        title = (
            r"Evolution of $\nu_m$ and $\nu_c$ (from spectrum, median jet parameters)"
        )

    fig = _figure_break_frequency_evolution(times, nu_m_all, nu_c_all, title)

    if save_plot:
        pdf_path = f"{basedir}/break_frequencies_evolution.pdf"
        png_path = f"{basedir}/break_frequencies_evolution.png"
        logging.info("Saving break-frequency plot to: %s", pdf_path)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        logging.info("Saving break-frequency plot to: %s", png_path)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


def _sanitize_filename_component(name: str) -> str:
    """Turn a filter label into a safe filename fragment."""
    out = []
    for c in name:
        if c.isalnum() or c in "-._":
            out.append(c)
        else:
            out.append("_")
    return "".join(out).strip("_") or "band"


def residual_plot(
    basedir,
    median_params,
    observed_data,
    filt,
    show_plot=False,
    save_plot=True,
):
    """
    Plot fractional residuals (observed − model) / model for one band vs time.

    Uses the same styling as lc_plot but does not draw posterior uncertainty ribbons.
    """
    plt.style.use(["science", "high-vis"])

    mpl.rcParams.update(
        {
            **_PAPER_SERIF_RC,
            "font.size": 5,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
        }
    )

    if filt not in filt_freqs:
        logger.warning(
            "residual_plot: unknown filter %r; known keys include %s",
            filt,
            sorted(filt_freqs.keys())[:10],
        )
        return
    if filt not in multipliers:
        logger.warning(
            "residual_plot: filter %r has no flux multiplier; skipping.", filt
        )
        return

    nu = filt_freqs[filt]
    multiplier = multipliers[filt]

    df_allobs = pd.read_csv(observed_data)
    df_allobs["Times"] = pd.to_numeric(df_allobs["Times"], errors="coerce")
    df_allobs["Fluxes"] = pd.to_numeric(df_allobs["Fluxes"], errors="coerce")
    df_allobs["FluxErrs"] = pd.to_numeric(df_allobs["FluxErrs"], errors="coerce")
    # Normalize UL column: treat missing/blank as "N"
    df_allobs = _normalize_ul_column(df_allobs)

    det = df_allobs[(df_allobs["Filt"] == filt) & (df_allobs["UL"] == "N")][
        ["Times", "Fluxes", "FluxErrs"]
    ].sort_values(by="Times")
    if len(det) == 0:
        logger.warning(
            "residual_plot: no detections (UL='N') for filter %r; skipping plot.", filt
        )
        return

    times = det["Times"].to_numpy()
    f_obs = det["Fluxes"].to_numpy()
    f_err = det["FluxErrs"].to_numpy()

    f_model = np.asarray(model(times, [nu], median_params))
    # Avoid divide-by-zero in pathological cases
    tiny = np.finfo(float).tiny
    denom = np.where(np.abs(f_model) > tiny, f_model, np.copysign(tiny, f_model + tiny))
    residual = (f_obs - f_model) / denom
    yerr = f_err / np.abs(denom)

    logger.info(
        "residual_plot: filt=%s, n_points=%s, residual range [%s, %s]",
        filt,
        len(times),
        float(np.nanmin(residual)),
        float(np.nanmax(residual)),
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.75, alpha=0.8)
    ax.errorbar(
        times,
        residual,
        yerr=yerr,
        fmt="o",
        markersize=4,
        alpha=1,
        color=band_colors.get(filt, "#616569"),
        mec="black",
        elinewidth=0.5,
        capsize=2,
        label=filt,
    )

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$(F_\mathrm{obs} - F_\mathrm{model}) / F_\mathrm{model}$")
    ax.set_title(f"Residuals: {filt} (×{multiplier} in LC plot)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(edgecolor="none", loc="best")
    fig.tight_layout()

    safe = _sanitize_filename_component(filt)
    if save_plot:
        pdf_path = f"{basedir}/residual_{safe}.pdf"
        png_path = f"{basedir}/residual_{safe}.png"
        logging.info("Saving residual plot to: %s", pdf_path)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        logging.info("Saving residual plot to: %s", png_path)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


################################################
def main():
    SAMPLE_USAGE = (
        "Example:\n"
        "  python jetsimpy_plot.py --obsfile data/GRB250916A_cons.csv  "
        " --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, "
        " n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'\n"
    )

    parser = ArgumentParser(
        description="Plot GRB afterglow model light curves using jetsimpy and overlay observations.",
        epilog=SAMPLE_USAGE,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def _error(message: str):
        parser.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        print("\n" + SAMPLE_USAGE, file=sys.stderr)
        sys.exit(2)

    parser.error = _error
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="a descriptive text label to identify this run",
    )
    parser.add_argument(
        "--obsfile",
        type=str,
        default="multinest_EP/mcmc_df_trunc.csv",
        help="csv file containing observed light curve",
    )
    # Will accept --params.key=value and build nested dicts
    parser.add_argument("--params", type=dict, default={})
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help=(
            "Plot afterglow spectrum F_nu vs frequency using Jet.Flux(); overlay "
            "detections from --obsfile at fixed epochs (see SPECTRUM_PLOT_TIME_EPOCHS, ±500 s)."
        ),
    )
    parser.add_argument(
        "--plot-break-frequencies",
        action="store_true",
        help=(
            "Plot evolution of synchrotron break frequencies nu_m and nu_c "
            "for the median parameters."
        ),
    )
    args = parser.parse_args()

    np.random.seed(12)

    # read obs csv
    file = args.obsfile

    # Set up the output directory and logging
    basedir = f"output"
    os.makedirs(basedir, exist_ok=True)
    outputfiles_basename = basedir + f"/jetsimpy_plot_"
    # Configure basic logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{outputfiles_basename}.log"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Commandline: {' '.join(sys.argv)}")

    # light curve fitting plot
    """
    params['jetType']=args.jetType
    params['z']=args.redshift

    params['jetType']=args.jetType
    params['loge0']=np.log10(4.87e52)
    params['logepsb']=np.log10(0.0448)
    params['logepse']=np.log10(0.3981)
    params['logn0']=np.log10(0.0032)
    params['thc']=0.0623
    params['thv']=0.0014
    params['p']=2.3578
    params['loglf']=100
    params['A']=0
    params['s']=0
    params['z']=args.redshift
    """
    ran_special_plot = False
    if args.spectrum:
        df_spectrum_obs = pd.read_csv(args.obsfile)
        epoch_obs = build_spectrum_epoch_observations(
            df_spectrum_obs, SPECTRUM_PLOT_TIME_EPOCHS, dt_sec=500.0
        )
        spectrum_plot(
            basedir,
            args.params,
            time_epochs=SPECTRUM_PLOT_TIME_EPOCHS,
            epoch_observations=epoch_obs,
        )
        ran_special_plot = True
    if args.plot_break_frequencies:
        break_frequency_evolution_plot(basedir, args.params)
        ran_special_plot = True
    if not ran_special_plot:
        lc_plot(basedir, args.params, [], args.obsfile)
    """
    params = {}
    params['jetType']=args.jetType
    params['z']=args.redshift

    params['jetType']=args.jetType
    params['loge0']=np.log10(1.25e52)
    params['logepsb']=np.log10(0.077)
    params['logepse']=np.log10(2.0189)
    params['logn0']=np.log10(0.0107)
    params['thc']=0.0759
    params['thv']=5.03e-4
    params['p']=2.0961
    params['loglf']=100
    params['A']=0
    params['s']=0
    params['z']=args.redshift
    lc_plot(params, observed_data=args.fullobsfile, observed_data_fit=args.obsfile, plotno=2)
    """


if __name__ == "__main__":
    main()
