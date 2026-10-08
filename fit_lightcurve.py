#!/usr/bin/env python3


"""
Fit GRB optical/NIR light curves with single, broken, or double-broken
power-law models.

Input CSV columns expected:
    Times
    Filt
    Freqs
    Fluxes
    FluxErrs
    UL

Example:
    python fit_lightcurve.py data/GRB260924A.csv --model single
        -> output/GRB260924A_<bands>_single.pdf

    python fit_lightcurve.py data/GRB260924A.csv --model broken \
        --bands g r z J
        -> output/GRB260924A_g_r_z_J_broken.pdf

    python fit_lightcurve.py data/GRB260924A.csv --model double \
        --break-epoch 1e5
        -> one break fixed at 1e5 s; the other break is free
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# Constants
# ============================================================

LN10 = np.log(10.0)


# ============================================================
# Power-law models
# ============================================================


def single_powerlaw(logt, logF0, alpha):
    """
    F = F0 * (t / 1 s)^(-alpha)

    In log10:
        logF = logF0 - alpha * logt
    """
    return logF0 - alpha * logt


def broken_powerlaw(logt, logF0, alpha1, alpha2, logtb):
    """
    Continuous broken power law.

    Before tb:
        F = F0 * t^(-alpha1)

    After tb:
        F = F(tb) * (t/tb)^(-alpha2)
    """

    yb = logF0 - alpha1 * logtb

    return np.where(logt <= logtb, logF0 - alpha1 * logt, yb - alpha2 * (logt - logtb))


def double_broken_powerlaw(logt, logF0, alpha1, alpha2, alpha3, logtb1, logtb2):
    """
    Continuous double-broken power law.

    alpha1 before tb1
    alpha2 between tb1 and tb2
    alpha3 after tb2
    """

    # Flux at first break
    yb1 = logF0 - alpha1 * logtb1

    # Flux at second break
    yb2 = yb1 - alpha2 * (logtb2 - logtb1)

    result = np.empty_like(logt)

    m1 = logt <= logtb1
    m2 = (logt > logtb1) & (logt <= logtb2)
    m3 = logt > logtb2

    result[m1] = logF0 - alpha1 * logt[m1]

    result[m2] = yb1 - alpha2 * (logt[m2] - logtb1)

    result[m3] = yb2 - alpha3 * (logt[m3] - logtb2)

    return result


def expand_params(params, model, logtb_fixed=None):
    """
    Insert a fixed break into the parameter vector used by the models.

    With no fixed break, params are already the full vector.

    Broken, fixed break:
        [logF0, alpha1, alpha2] -> [..., logtb_fixed]

    Double broken, fixed break:
        [logF0, alpha1, alpha2, alpha3, logtb_free]
        -> breaks ordered so tb1 <= tb2, one of them equal to the epoch.
    """

    params = np.asarray(params, dtype=float)

    if logtb_fixed is None:
        return params

    if model == "broken":
        return np.array([params[0], params[1], params[2], logtb_fixed])

    if model == "double":
        logtb1, logtb2 = np.sort([logtb_fixed, params[4]])

        return np.array(
            [params[0], params[1], params[2], params[3], logtb1, logtb2]
        )

    raise ValueError(f"A fixed break is not used with model: {model}")


def fixed_break_index(full_params, logtb_fixed):
    """Index of the break parameter that matches the forced epoch."""

    if len(full_params) == 4:
        return 3

    if abs(full_params[4] - logtb_fixed) <= abs(full_params[5] - logtb_fixed):
        return 4

    return 5


# ============================================================
# Initial guesses
# ============================================================


def initial_guess(logt, logf, model, logtb_fixed=None):
    """
    Construct robust initial guesses.

    The guesses are deliberately conservative because the optimizer
    is working in log-space.
    """

    logt_min = np.min(logt)
    logt_max = np.max(logt)

    logf_med = np.median(logf)

    # --------------------------------------------------------
    # Single power law
    # --------------------------------------------------------

    if model == "single":
        # Estimate slope using endpoints
        if len(logt) >= 2 and np.ptp(logt) > 0:
            alpha = -((logf[-1] - logf[0]) / (logt[-1] - logt[0]))

        else:
            alpha = 1.0

        alpha = np.clip(alpha, -5.0, 5.0)

        # Estimate normalization
        logF0 = np.median(logf + alpha * logt)

        return np.array([logF0, alpha])

    # --------------------------------------------------------
    # Broken power law
    # --------------------------------------------------------

    elif model == "broken":
        # Break near middle of time range, unless one epoch is forced.
        logtb = np.median(logt) if logtb_fixed is None else logtb_fixed

        # Estimate slopes independently on either side
        before = logt <= logtb
        after = logt > logtb

        if np.sum(before) >= 2:
            alpha1 = -np.polyfit(logt[before], logf[before], 1)[0]

        else:
            alpha1 = 1.0

        if np.sum(after) >= 2:
            alpha2 = -np.polyfit(logt[after], logf[after], 1)[0]

        else:
            alpha2 = alpha1

        alpha1 = np.clip(alpha1, -5.0, 5.0)
        alpha2 = np.clip(alpha2, -5.0, 5.0)

        logF0 = np.median(logf + alpha1 * logt)

        if logtb_fixed is not None:
            return np.array([logF0, alpha1, alpha2])

        return np.array([logF0, alpha1, alpha2, logtb])

    # --------------------------------------------------------
    # Double broken power law
    # --------------------------------------------------------

    elif model == "double":
        if logtb_fixed is None:
            # Divide the logarithmic time range into thirds
            logtb1 = logt_min + (logt_max - logt_min) / 3.0
            logtb2 = logt_min + 2.0 * (logt_max - logt_min) / 3.0

        else:
            # Put the free break on the side of the epoch with more data.
            left_span = logtb_fixed - logt_min
            right_span = logt_max - logtb_fixed

            if right_span >= left_span:
                logtb_free = logtb_fixed + 0.5 * max(right_span, 0.1)

            else:
                logtb_free = logtb_fixed - 0.5 * max(left_span, 0.1)

            logtb1, logtb2 = np.sort([logtb_fixed, logtb_free])

        before = logt <= logtb1
        middle = (logt > logtb1) & (logt <= logtb2)
        after = logt > logtb2

        # First slope
        if np.sum(before) >= 2:
            alpha1 = -np.polyfit(logt[before], logf[before], 1)[0]

        else:
            alpha1 = 1.0

        # Middle slope
        if np.sum(middle) >= 2:
            alpha2 = -np.polyfit(logt[middle], logf[middle], 1)[0]

        else:
            alpha2 = alpha1

        # Final slope
        if np.sum(after) >= 2:
            alpha3 = -np.polyfit(logt[after], logf[after], 1)[0]

        else:
            alpha3 = alpha2

        alpha1 = np.clip(alpha1, -5.0, 5.0)
        alpha2 = np.clip(alpha2, -5.0, 5.0)
        alpha3 = np.clip(alpha3, -5.0, 5.0)

        logF0 = np.median(logf + alpha1 * logt)

        if logtb_fixed is not None:
            logtb_free = logtb2 if np.isclose(logtb1, logtb_fixed) else logtb1

            return np.array([logF0, alpha1, alpha2, alpha3, logtb_free])

        return np.array([logF0, alpha1, alpha2, alpha3, logtb1, logtb2])

    else:
        raise ValueError(f"Unknown model: {model}")


# ============================================================
# Model evaluation
# ============================================================


def evaluate_model(logt, params, model):

    if model == "single":
        return single_powerlaw(logt, *params)

    elif model == "broken":
        return broken_powerlaw(logt, *params)

    elif model == "double":
        return double_broken_powerlaw(logt, *params)

    else:
        raise ValueError(f"Unknown model: {model}")


# ============================================================
# Residuals
# ============================================================


def residuals(params, logt, logf, sigma_logf, model, logtb_fixed=None):

    params = expand_params(params, model, logtb_fixed)

    # Explicitly reject invalid double-break ordering
    if model == "double":
        logtb1 = params[4]
        logtb2 = params[5]

        if logtb2 <= logtb1:
            # Return a large penalty rather than allowing an
            # unphysical model.
            return np.ones_like(logf) * 1e6

    model_logf = evaluate_model(logt, params, model)

    return (logf - model_logf) / sigma_logf


# ============================================================
# Parameter bounds
# ============================================================


def parameter_bounds(logt, logf, model, logtb_fixed=None):

    logt_min = np.min(logt)
    logt_max = np.max(logt)

    # Allow a small amount outside the observed time range.
    padding = 0.02 * max(logt_max - logt_min, 1.0)

    lower_tb = logt_min - padding
    upper_tb = logt_max + padding

    # Normalization bounds
    lower_logF = np.min(logf) - 10.0
    upper_logF = np.max(logf) + 10.0

    if model == "single":
        lower = np.array([lower_logF, -10.0])

        upper = np.array([upper_logF, 10.0])

    elif model == "broken":
        if logtb_fixed is None:
            lower = np.array([lower_logF, -10.0, -10.0, lower_tb])

            upper = np.array([upper_logF, 10.0, 10.0, upper_tb])

        else:
            lower = np.array([lower_logF, -10.0, -10.0])

            upper = np.array([upper_logF, 10.0, 10.0])

    elif model == "double":
        if logtb_fixed is None:
            lower = np.array([lower_logF, -10.0, -10.0, -10.0, lower_tb, lower_tb])

            upper = np.array([upper_logF, 10.0, 10.0, 10.0, upper_tb, upper_tb])

        else:
            lower = np.array([lower_logF, -10.0, -10.0, -10.0, lower_tb])

            upper = np.array([upper_logF, 10.0, 10.0, 10.0, upper_tb])

    else:
        raise ValueError(f"Unknown model: {model}")

    return lower, upper


# ============================================================
# Make initial guess feasible
# ============================================================


def make_feasible_initial_guess(p0, lower, upper, model, logtb_fixed=None):

    p0 = np.asarray(p0, dtype=float).copy()

    # --------------------------------------------------------
    # General clipping
    # --------------------------------------------------------

    eps = 1e-8

    p0 = np.maximum(p0, lower + eps)

    p0 = np.minimum(p0, upper - eps)

    # --------------------------------------------------------
    # Explicitly enforce break ordering
    # --------------------------------------------------------

    if model == "double" and logtb_fixed is not None:
        # Keep the free break away from the forced epoch so the
        # middle segment is not degenerate at the start.
        if abs(p0[4] - logtb_fixed) < 1e-4:
            if (logtb_fixed - lower[4]) >= (upper[4] - logtb_fixed):
                p0[4] = logtb_fixed - 0.2 * (logtb_fixed - lower[4])

            else:
                p0[4] = logtb_fixed + 0.2 * (upper[4] - logtb_fixed)

        p0[4] = np.clip(p0[4], lower[4] + eps, upper[4] - eps)

    elif model == "double":
        tb1 = p0[4]
        tb2 = p0[5]

        # If ordering is invalid, reset them to sensible positions.
        if tb2 <= tb1:
            mid = 0.5 * (lower[4] + upper[4])

            span = 0.2 * (upper[4] - lower[4])

            tb1 = mid - span
            tb2 = mid + span

        # Make sure both remain within bounds
        tb1 = np.clip(tb1, lower[4] + eps, upper[4] - eps)

        tb2 = np.clip(tb2, lower[5] + eps, upper[5] - eps)

        # Final ordering check
        if tb2 <= tb1:
            tb1 = lower[4] + 0.3 * (upper[4] - lower[4])

            tb2 = lower[5] + 0.7 * (upper[5] - lower[5])

        p0[4] = tb1
        p0[5] = tb2

    return p0


# ============================================================
# Minimum number of observations
# ============================================================


def minimum_points(model):

    if model == "single":
        return 2

    elif model == "broken":
        return 3

    elif model == "double":
        return 4

    else:
        raise ValueError(f"Unknown model: {model}")


# ============================================================
# Fit a single band
# ============================================================


def fit_band(df_band, model, break_epoch=None):

    # --------------------------------------------------------
    # Remove upper limits from fit
    # --------------------------------------------------------

    detections = df_band[
        ~df_band["UL"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
    ].copy()

    # Require positive values
    detections = detections[
        (detections["Times"] > 0)
        & (detections["Fluxes"] > 0)
        & (detections["FluxErrs"] > 0)
    ].copy()

    n = len(detections)

    nmin = minimum_points(model)

    if n < nmin:
        print(f"  Not enough detections for {model}: {n} < {nmin}")

        return None

    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    t = detections["Times"].values.astype(float)
    f = detections["Fluxes"].values.astype(float)
    ferr = detections["FluxErrs"].values.astype(float)

    # Sort by time
    idx = np.argsort(t)

    t = t[idx]
    f = f[idx]
    ferr = ferr[idx]

    # --------------------------------------------------------
    # Convert to log10
    # --------------------------------------------------------

    logt = np.log10(t)
    logf = np.log10(f)

    logtb_fixed = None

    if break_epoch is not None:
        logtb_fixed = np.log10(break_epoch)

        if break_epoch < t.min() or break_epoch > t.max():
            print(
                f"  Fixed break at {break_epoch:.3g} s is outside the "
                f"detection range {t.min():.3g}–{t.max():.3g} s"
            )

    # Propagation:
    #
    # sigma(log10 F) = sigma_F / (F ln 10)
    #
    sigma_logf = ferr / (f * LN10)

    # Prevent zero / absurdly tiny uncertainties
    sigma_logf = np.maximum(sigma_logf, 1e-5)

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    p0 = initial_guess(logt, logf, model, logtb_fixed)

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    lower, upper = parameter_bounds(logt, logf, model, logtb_fixed)

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Ensure x0 lies inside bounds.
    # --------------------------------------------------------

    p0 = make_feasible_initial_guess(p0, lower, upper, model, logtb_fixed)

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    result = least_squares(
        residuals,
        p0,
        args=(logt, logf, sigma_logf, model, logtb_fixed),
        bounds=(lower, upper),
        max_nfev=10000,
        loss="linear",
    )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    best_free = result.x

    best_params = expand_params(best_free, model, logtb_fixed)

    res = residuals(best_free, logt, logf, sigma_logf, model, logtb_fixed)

    chi2 = np.sum(res**2)

    k = len(best_free)

    dof = n - k

    if dof > 0:
        reduced_chi2 = chi2 / dof
    else:
        reduced_chi2 = np.nan

    bic = chi2 + k * np.log(n)

    # --------------------------------------------------------
    # Covariance estimate
    # --------------------------------------------------------

    covariance = None

    try:
        jac = result.jac

        jtj = jac.T @ jac

        covariance = np.linalg.inv(jtj) * reduced_chi2

        errors_free = np.sqrt(np.diag(covariance))

        errors = _embed_free_errors(
            errors_free, model, best_params, logtb_fixed
        )

        covariance = _embed_free_covariance(
            covariance, model, best_params, logtb_fixed
        )

    except np.linalg.LinAlgError:
        errors = np.full(len(best_params), np.nan)

    return {
        "params": best_params,
        "errors": errors,
        "covariance": covariance,
        "result": result,
        "t": t,
        "f": f,
        "ferr": ferr,
        "logt": logt,
        "logf": logf,
        "sigma_logf": sigma_logf,
        "residuals": res,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "bic": bic,
        "n": n,
        "model": model,
        "break_epoch": break_epoch,
        "logtb_fixed": logtb_fixed,
    }


def _free_parameter_indices(model, full_params, logtb_fixed):
    """Positions in the full parameter vector that were optimized."""

    if logtb_fixed is None:
        return np.arange(len(full_params))

    if model == "broken":
        return np.array([0, 1, 2])

    fixed_at = fixed_break_index(full_params, logtb_fixed)

    free = [0, 1, 2, 3, 4 if fixed_at == 5 else 5]

    return np.array(free)


def _embed_free_errors(errors_free, model, full_params, logtb_fixed):

    errors = np.zeros(len(full_params))

    errors[_free_parameter_indices(model, full_params, logtb_fixed)] = errors_free

    return errors


def _embed_free_covariance(covariance_free, model, full_params, logtb_fixed):

    n = len(full_params)

    covariance = np.zeros((n, n))

    idx = _free_parameter_indices(model, full_params, logtb_fixed)

    covariance[np.ix_(idx, idx)] = covariance_free

    return covariance


# ============================================================
# Print fit result
# ============================================================


def print_fit_result(band, fit):

    if fit is None:
        return

    p = fit["params"]
    e = fit["errors"]

    print()
    print("=" * 70)
    print(f"Band: {band}")
    print("=" * 70)

    print(
        f"N = {fit['n']}, "
        f"chi2 = {fit['chi2']:.3f}, "
        f"dof = {fit['dof']}, "
        f"reduced chi2 = {fit['reduced_chi2']:.3f}, "
        f"BIC = {fit['bic']:.3f}"
    )

    if fit["model"] == "single":
        print(f"logF0 = {p[0]:.5f} +/- {e[0]:.5f}")

        print(f"alpha = {p[1]:.5f} +/- {e[1]:.5f}")

    elif fit["model"] == "broken":
        print(f"logF0 = {p[0]:.5f} +/- {e[0]:.5f}")

        print(f"alpha1 = {p[1]:.5f} +/- {e[1]:.5f}")

        print(f"alpha2 = {p[2]:.5f} +/- {e[2]:.5f}")

        tb_fixed = fit.get("logtb_fixed") is not None

        print(f"log10(tb/s) = {_format_value(p[3], e[3], tb_fixed)}")

        print(f"tb = {10 ** p[3]:.3g} s")

    elif fit["model"] == "double":
        print(f"logF0 = {p[0]:.5f} +/- {e[0]:.5f}")

        print(f"alpha1 = {p[1]:.5f} +/- {e[1]:.5f}")

        print(f"alpha2 = {p[2]:.5f} +/- {e[2]:.5f}")

        print(f"alpha3 = {p[3]:.5f} +/- {e[3]:.5f}")

        fixed_at = None

        if fit.get("logtb_fixed") is not None:
            fixed_at = fixed_break_index(p, fit["logtb_fixed"])

        print(f"log10(tb1/s) = {_format_value(p[4], e[4], fixed_at == 4)}")

        print(f"log10(tb2/s) = {_format_value(p[5], e[5], fixed_at == 5)}")

        print(f"tb1 = {10 ** p[4]:.3g} s")

        print(f"tb2 = {10 ** p[5]:.3g} s")


def _format_value(value, error, fixed):

    if fixed:
        return f"{value:.5f} (fixed)"

    return f"{value:.5f} +/- {error:.5f}"


# ============================================================
# Plot
# ============================================================


def powerlaw_sections(tmin, tmax, params, model):
    """
    Time intervals and temporal index for each power-law segment.

    Returns (t_left, t_right, alpha), clipped to the plotted time range.
    """

    if model == "single":
        breaks = []
        alphas = [params[1]]

    elif model == "broken":
        breaks = [10 ** params[3]]
        alphas = [params[1], params[2]]

    elif model == "double":
        breaks = [10 ** params[4], 10 ** params[5]]
        alphas = [params[1], params[2], params[3]]

    else:
        return []

    edges = [tmin, *breaks, tmax]
    sections = []

    for left, right, alpha in zip(edges[:-1], edges[1:], alphas):
        left = max(float(left), tmin)
        right = min(float(right), tmax)

        if right > left:
            sections.append((left, right, float(alpha)))

    return sections


def label_powerlaw_slopes(ax, tmin, tmax, params, model, color):
    """Write the fitted temporal index at the middle of each segment."""

    sections = powerlaw_sections(tmin, tmax, params, model)
    span = np.log10(tmax) - np.log10(tmin)

    for t_left, t_right, alpha in sections:
        segment = np.log10(t_right) - np.log10(t_left)

        # A single power law fills the panel even when the data span is
        # short. Only drop a piece of a broken power law that is too
        # narrow to read next to its neighbours.
        if len(sections) > 1 and span > 0 and segment / span < 0.12:
            continue

        t_mid = np.sqrt(t_left * t_right)
        f_mid = 10 ** evaluate_model(np.log10([t_mid]), params, model)[0]

        ax.annotate(
            rf"$\alpha={alpha:.2f}$",
            xy=(t_mid, f_mid),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=11,
            annotation_clip=False,
            clip_on=False,
            zorder=5,
        )


def mark_break(ax, axres, tb):
    """Draw a break and label it with the epoch in seconds."""

    ax.axvline(tb, linestyle=":", linewidth=1.5)

    axres.axvline(tb, linestyle=":", linewidth=1.5)

    ax.text(
        tb,
        0.98,
        f"{tb:.3g} s",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        clip_on=False,
        zorder=6,
    )


def plot_results(data, fits, model, bands, output=None):

    # --------------------------------------------------------
    # Figure
    #
    # Each band is a light-curve panel with its residual panel
    # attached underneath (no gap). Bands are separated from
    # each other.
    # --------------------------------------------------------

    fig = plt.figure(figsize=(10, 3.8 * len(bands)))

    outer = fig.add_gridspec(
        len(bands),
        1,
        left=0.10,
        right=0.97,
        top=0.97,
        bottom=0.07,
        hspace=0.32,
    )

    # --------------------------------------------------------
    # Shared time range
    # --------------------------------------------------------

    plotted = data[data["Filt"].astype(str).isin(bands)]
    times = plotted.loc[plotted["Times"] > 0, "Times"]
    t_lo = float(times.min())
    t_hi = float(times.max())

    # --------------------------------------------------------
    # Shared flux range
    # --------------------------------------------------------

    fluxes = plotted.loc[
        (plotted["Times"] > 0) & (plotted["Fluxes"] > 0),
        "Fluxes",
    ]
    f_lo = float(fluxes.min())
    f_hi = float(fluxes.max())

    # --------------------------------------------------------
    # Plot each band
    # --------------------------------------------------------

    for i, band in enumerate(bands):
        inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)

        ax = fig.add_subplot(inner[0])
        axres = fig.add_subplot(inner[1], sharex=ax)

        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        df_band = data[data["Filt"].astype(str) == band].copy()

        # ----------------------------------------------------
        # Detections
        # ----------------------------------------------------

        detections = df_band[
            ~df_band["UL"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
        ].copy()

        detections = detections[(detections["Times"] > 0) & (detections["Fluxes"] > 0)]

        if len(detections) > 0:
            ax.errorbar(
                detections["Times"],
                detections["Fluxes"],
                yerr=detections["FluxErrs"],
                fmt="o",
                capsize=3,
                label=f"{band} detections",
            )

        # ----------------------------------------------------
        # Upper limits
        # ----------------------------------------------------

        upper = df_band[
            df_band["UL"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
        ].copy()

        upper = upper[(upper["Times"] > 0) & (upper["Fluxes"] > 0)]

        if len(upper) > 0:
            ax.errorbar(
                upper["Times"],
                upper["Fluxes"],
                yerr=0.15 * upper["Fluxes"],
                uplims=True,
                fmt="v",
                markersize=6,
                label=f"{band} upper limits",
            )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        fit = fits.get(band)

        if fit is not None:
            tmin = np.min(df_band["Times"][df_band["Times"] > 0])

            tmax = np.max(df_band["Times"][df_band["Times"] > 0])

            tmodel = np.logspace(np.log10(tmin), np.log10(tmax), 500)

            logtmodel = np.log10(tmodel)

            logfmodel = evaluate_model(logtmodel, fit["params"], model)

            fmodel = 10**logfmodel

            (fit_line,) = ax.plot(
                tmodel, fmodel, "-", linewidth=2, label=f"{model} fit"
            )

            fit_color = fit_line.get_color()

            # ------------------------------------------------
            # Residuals
            # ------------------------------------------------

            tres = fit["t"]

            residuals_sigma = fit["residuals"]

            axres.axhline(0, linestyle="--", linewidth=1)

            axres.errorbar(tres, residuals_sigma, fmt="o")

            axres.set_ylabel(r"Residual ($\sigma$)")

            axres.set_ylim(
                min(-4, np.min(residuals_sigma) - 1),
                max(4, np.max(residuals_sigma) + 1),
            )

        # ----------------------------------------------------
        # Main plot formatting
        # ----------------------------------------------------

        ax.set_xscale("log")
        ax.set_yscale("log")

        if fit is not None:
            label_powerlaw_slopes(ax, tmin, tmax, fit["params"], model, fit_color)

        ax.set_ylabel("Flux density (mJy)")

        ax.legend(loc="best")

        ax.grid(True, which="both", alpha=0.3)

        axres.set_xscale("log")

        axres.grid(True, which="both", alpha=0.3)

        # ----------------------------------------------------
        # Statistics in plot
        # ----------------------------------------------------

        if fit is not None:
            text = (
                rf"$\chi^2_\nu={fit['reduced_chi2']:.2f}$"
                "\n"
                rf"BIC = {fit['bic']:.1f}"
            )

            ax.text(
                0.03,
                0.05,
                text,
                transform=ax.transAxes,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", alpha=0.8),
            )

        # ----------------------------------------------------
        # Break lines
        # ----------------------------------------------------

        if fit is not None:
            p = fit["params"]

            if model == "broken":
                mark_break(ax, axres, 10 ** p[3])

            elif model == "double":
                mark_break(ax, axres, 10 ** p[4])
                mark_break(ax, axres, 10 ** p[5])

        # ----------------------------------------------------
        # X label and shared limits
        # ----------------------------------------------------

        axres.set_xlabel("Time since trigger (s)")

        if len(bands) > 1:
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(f_lo, f_hi)

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight", format="pdf")

        print(f"\nSaved figure to: {output}")

    else:
        plt.show()


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fit GRB light curves with single, broken, or double-broken power laws."
        )
    )

    parser.add_argument("input", help="Input CSV file")

    parser.add_argument(
        "--model",
        choices=["single", "broken", "double"],
        default="single",
        help="Model to fit",
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=("Filters to fit. If omitted, all filters are used."),
    )

    parser.add_argument(
        "--break-epoch",
        type=float,
        default=None,
        help=(
            "Force one break at this time (seconds since trigger). "
            "Used with --model broken (the only break) or double "
            "(one break fixed, the other free)."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output figure filename. If omitted, written to "
            "output/<data>_<bands>_<model>.pdf"
        ),
    )

    args = parser.parse_args()

    if args.break_epoch is not None:
        if args.model not in ("broken", "double"):
            parser.error("--break-epoch requires --model broken or double")

        if args.break_epoch <= 0:
            parser.error("--break-epoch must be positive (seconds since trigger)")

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    data = pd.read_csv(args.input)

    required_columns = ["Times", "Filt", "Freqs", "Fluxes", "FluxErrs", "UL"]

    missing = [c for c in required_columns if c not in data.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # --------------------------------------------------------
    # Clean basic data
    # --------------------------------------------------------

    data["Times"] = pd.to_numeric(data["Times"], errors="coerce")

    data["Fluxes"] = pd.to_numeric(data["Fluxes"], errors="coerce")

    data["FluxErrs"] = pd.to_numeric(data["FluxErrs"], errors="coerce")

    data = data.dropna(subset=["Times", "Fluxes"])

    # --------------------------------------------------------
    # Determine bands
    # --------------------------------------------------------

    if args.bands is None:
        bands = sorted(data["Filt"].astype(str).unique())

    else:
        bands = args.bands

    print()
    print("=" * 70)
    print("GRB LIGHT-CURVE FIT")
    print("=" * 70)

    print(f"Input : {args.input}")

    print(f"Model : {args.model}")

    print(f"Bands : {', '.join(bands)}")

    if args.break_epoch is not None:
        print(f"Break : {args.break_epoch:.6g} s (fixed)")

    if args.output is None:
        stem = Path(args.input).stem
        band_part = "_".join(bands)
        output = Path("output") / f"{stem}_{band_part}_{args.model}.pdf"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output = str(output)

    print(f"Output: {output}")

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fits = {}

    for band in bands:
        print()
        print(f"Fitting band: {band}")

        df_band = data[data["Filt"].astype(str) == band].copy()

        fit = fit_band(df_band, args.model, args.break_epoch)

        fits[band] = fit

        if fit is not None:
            print_fit_result(band, fit)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    if bands:
        plot_results(data, fits, args.model, bands, output)
    else:
        print("\nNo requested plot bands were fitted; figure not written.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
