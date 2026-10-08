# afterglow_plot.py — Plot GRB afterglow light curves from saved model output
#
# Description:
# - Draws figures from a plot-data file produced by afterglow_model.compute_plot_data.
# - Does not evaluate the afterglow model.
#
# CLI:
#   --obsfile  Path to observations CSV (used only to build the plot-data file).
#   --params   JSON/dict of model parameters; include `jetType` and `z`.
#   --library  Afterglow library: jetsimpy (default) or vegas.
#   --spectrum Plot the saved F_nu vs nu curves.
#   --label    Optional label (reserved).
#
# Outputs:
#   output/plot_data.pkl
#   output/lc_afterflow_obs_matching.pdf and .png
#   output/afterglow_plot_.log
#
# Example:
#   python afterglow_plot.py --obsfile data/GRB250916A_cons.csv  \
# --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, \
# n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'

import os
import random
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


import numpy as np
import logging
import argparse
from lc_plot_settings import lc_plot_settings_default

######################

logger = logging.getLogger(__name__)

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


def _color_for_band(band, color_map):
    """Return the mapped color, or a stable random color when the band is unmapped."""
    if band in color_map:
        return color_map[band]
    rng = random.Random(band)
    color = "#{:06x}".format(rng.randint(0, 0xFFFFFF))
    logger.info("No color mapping for band %r; using %s.", band, color)
    return color


def _multiplier_for_band(band, multipliers):
    """Return the mapped flux multiplier, or 1 when the band is unmapped."""
    if band in multipliers:
        return multipliers[band]
    logger.info("No multiplier for band %r; using 1.", band)
    return 1


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
    "VT_R": 3.63385e14,
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


def _apply_paper_style(legend_fontsize=12, line_width=0.75):
    plt.style.use(["science", "high-vis"])
    mpl.rcParams.update(
        {
            **_PAPER_SERIF_RC,
            "font.size": 5,  # minimum allowed by Nature
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": legend_fontsize,
            "pdf.fonttype": 42,  # embed fonts as TrueType
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": line_width,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
        }
    )


def _light_curve_style(plot_settings):
    settings = lc_plot_settings_default if plot_settings is None else plot_settings
    required_settings = {
        "xlim",
        "ylim",
        "multipliers",
        "band_colors",
        "band_secondary_colors",
    }
    missing_settings = required_settings.difference(settings)
    if missing_settings:
        raise ValueError(
            "plot_settings is missing required keys: "
            + ", ".join(sorted(missing_settings))
        )
    return settings


def lc_plot(
    basedir,
    plot_data,
    show_plot=False,
    save_plot=True,
    hide_z_text=False,
    plot_settings=None,
):
    """Plot a light curve from arrays in ``plot_data`` (no model evaluation)."""
    settings = _light_curve_style(plot_settings)
    xlim = settings["xlim"]
    ylim = settings["ylim"]
    multipliers = settings["multipliers"]
    band_colors = settings["band_colors"]
    band_secondary_colors = settings["band_secondary_colors"]
    _apply_paper_style()

    median_params = plot_data["median_params"]
    t = np.asarray(plot_data["lc_times"], dtype=float)
    df_allobs = plot_data["observations"]
    light_curves = plot_data["light_curves"]
    bands = [curve["band"] for curve in light_curves]
    band_multipliers = {
        band: _multiplier_for_band(band, multipliers) for band in bands
    }
    resolved_band_colors = {
        band: _color_for_band(band, band_colors) for band in bands
    }
    resolved_secondary_colors = {
        band: band_secondary_colors.get(band, resolved_band_colors[band])
        for band in bands
    }
    logger.info(
        "lc_plot: n_bands=%d, n_obs=%d, n_times=%d",
        len(light_curves),
        len(df_allobs),
        len(t),
    )

    logger.info("band,frequency_hz,time_days,time_seconds,flux_mjy")
    for row in plot_data["checkpoints"]:
        logger.info(
            "%s,%s,%s,%s,%s",
            row["band"],
            row["nu"],
            row["time_days"],
            row["time_seconds"],
            row["flux_mjy"],
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot the model curves saved from jetsimpy
    for curve in light_curves:
        band = curve["band"]
        multiplier = band_multipliers[band]
        nu = curve["nu"]
        logger.info(f"Plotting saved model for frequency: {nu}")
        Fnu_model = np.asarray(curve["median_flux"], dtype=float)
        if not np.any(np.isfinite(Fnu_model)):
            logger.error("saved median light curve for band %s is missing", band)
            continue

        sample_fluxes = np.asarray(curve["sample_fluxes"], dtype=float)
        for Fnu_sig3 in sample_fluxes:
            if not np.any(np.isfinite(Fnu_sig3)):
                continue
            ax.plot(
                t,
                Fnu_sig3 * multiplier,
                linewidth=1.0,
                linestyle="-",
                color=resolved_secondary_colors[band],
                alpha=0.2,
            )

        ax.plot(
            t,
            Fnu_model * multiplier,
            linewidth=1.0,
            linestyle="-",
            label=f"{band} x {multiplier}",
            color=resolved_band_colors[band],
            alpha=1,
        )

    # plot the actual observations
    for curve in light_curves:
        band = curve["band"]
        multiplier = band_multipliers[band]

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
                color=resolved_band_colors[band],
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
                color=resolved_band_colors[band],
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


# Observer-time unit used only to label spectrum epochs.
_SEC_PER_DAY = 86400.0


def _end_spectral_slope(nu, fnu, i0, i1):
    """Power-law index β in F_ν ∝ ν^β between two frequency samples."""
    nu0, nu1 = float(nu[i0]), float(nu[i1])
    f0, f1 = float(fnu[i0]), float(fnu[i1])
    if not (nu0 > 0 and nu1 > 0 and f0 > 0 and f1 > 0 and nu0 != nu1):
        return None
    beta = np.log(f1 / f0) / np.log(nu1 / nu0)
    return np.sqrt(nu0 * nu1), np.sqrt(f0 * f1), beta


def _label_spectrum_end_slopes(ax, nu, fnu, color):
    """Label β at the two lowest frequencies and the two highest frequencies."""
    for i0, i1 in ((0, 1), (-2, -1)):
        marked = _end_spectral_slope(nu, fnu, i0, i1)
        if marked is None:
            continue
        nu_mid, f_mid, beta = marked
        ax.annotate(
            rf"$\beta={beta:.2f}$",
            xy=(nu_mid, f_mid),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
            annotation_clip=True,
            zorder=6,
        )


def spectrum_plot(
    basedir,
    plot_data,
    show_plot=False,
    save_plot=True,
):
    """Plot the saved afterglow spectrum F_ν vs ν. Does not evaluate the model."""
    _apply_paper_style(legend_fontsize=10)

    spectrum = plot_data["spectrum"]
    times = np.asarray(spectrum["times"], dtype=float)
    nu_c = np.asarray(spectrum["nu"], dtype=float)
    fnu_all = np.asarray(spectrum["fnu"], dtype=float)
    obs_by_epoch = spectrum["observations"]
    n_ep = len(times)
    if len(obs_by_epoch) != n_ep:
        raise ValueError(
            f"spectrum observations must have length {n_ep}, got {len(obs_by_epoch)}"
        )
    if fnu_all.shape != (n_ep, len(nu_c)):
        raise ValueError(
            f"spectrum fnu shape {fnu_all.shape} does not match "
            f"({n_ep}, {len(nu_c)})"
        )

    colors = mpl.cm.viridis(np.linspace(0.15, 0.95, n_ep))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for it, t_obs in enumerate(times):
        ax.plot(
            nu_c,
            fnu_all[it],
            color=colors[it],
            linestyle="-",
            label=f"{t_obs / _SEC_PER_DAY:.4g} d",
        )
        _label_spectrum_end_slopes(ax, nu_c, fnu_all[it], colors[it])

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
    plot_data,
    show_plot=False,
    save_plot=True,
):
    """Plot saved synchrotron break frequencies. Does not evaluate the model."""
    _apply_paper_style(legend_fontsize=10, line_width=0.9)

    breaks = plot_data["breaks"]
    times = np.asarray(breaks["times"], dtype=float)
    nu_m_all = np.asarray(breaks["nu_m"], dtype=float)
    nu_c_all = np.asarray(breaks["nu_c"], dtype=float)
    if breaks["method"] == "analytical":
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
    plot_data,
    filt,
    show_plot=False,
    save_plot=True,
    plot_settings=None,
):
    """
    Plot fractional residuals (observed − model) / model for one band vs time.

    Model fluxes come from ``plot_data``; this function does not evaluate the model.
    """
    settings = _light_curve_style(plot_settings)
    xlim = settings["xlim"]
    _apply_paper_style()

    df_allobs = plot_data["observations"]
    known = set(df_allobs["Filt"].astype(str))
    if filt not in known:
        logger.warning(
            "residual_plot: filter %r is not in the saved observations", filt
        )
        return

    multiplier = _multiplier_for_band(filt, settings["multipliers"])
    color = _color_for_band(filt, settings["band_colors"])

    det = df_allobs[(df_allobs["Filt"] == filt) & (df_allobs["UL"] == "N")][
        ["Times", "Fluxes", "FluxErrs", "ModelFlux"]
    ].sort_values(by="Times")
    if len(det) == 0:
        logger.warning(
            "residual_plot: no detections (UL='N') for filter %r; skipping plot.", filt
        )
        return

    times = det["Times"].to_numpy()
    f_obs = det["Fluxes"].to_numpy()
    f_err = det["FluxErrs"].to_numpy()
    f_model = det["ModelFlux"].to_numpy(dtype=float)

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
        color=color,
        mec="black",
        elinewidth=0.5,
        capsize=2,
        label=filt,
    )

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
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
    from jsonargparse import ArgumentParser
    from afterglow_model import (
        PLOT_DATA_FILENAME,
        add_afterglow_library_argument,
        compute_plot_data,
        save_plot_data,
        set_afterglow_library,
    )

    SAMPLE_USAGE = (
        "Example:\n"
        "  python afterglow_plot.py --obsfile data/GRB250916A_cons.csv  "
        " --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, "
        " n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'\n"
    )

    parser = ArgumentParser(
        description="Evaluate the afterglow model, save plot data, then draw figures from that file.",
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
    add_afterglow_library_argument(parser)
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help=(
            "Plot the saved afterglow spectrum F_nu vs frequency and overlay "
            "detections from --obsfile at the stored epochs."
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
    set_afterglow_library(args.library)

    np.random.seed(12)

    # Set up the output directory and logging
    basedir = f"output"
    os.makedirs(basedir, exist_ok=True)
    outputfiles_basename = basedir + f"/afterglow_plot_"
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
    logger.info("Afterglow library: %s", args.library)
    plot_data = compute_plot_data(
        args.params,
        [],
        args.obsfile,
        lc_plot_settings_default["xlim"],
        library=args.library,
    )
    plot_data_path = os.path.join(basedir, PLOT_DATA_FILENAME)
    save_plot_data(plot_data_path, plot_data)
    logger.info("Saved plot data to %s", plot_data_path)

    ran_special_plot = False
    if args.spectrum:
        spectrum_plot(basedir, plot_data)
        ran_special_plot = True
    if args.plot_break_frequencies:
        break_frequency_evolution_plot(basedir, plot_data)
        ran_special_plot = True
    if not ran_special_plot:
        lc_plot(basedir, plot_data)
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
