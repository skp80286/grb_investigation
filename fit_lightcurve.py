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

--common fits one temporal power law to several bands at once. The
indices and break times are shared. Each band has its own normalization.
Bands omitted from the fit are still plotted on that same shape.

Example:
    python fit_lightcurve.py data/GRB260924A.csv --model single
        -> output/GRB260924A_<bands>_single.pdf

    python fit_lightcurve.py data/GRB260924A.csv --model broken \
        --bands g r z J
        -> output/GRB260924A_g_r_z_J_broken.pdf

    python fit_lightcurve.py data/GRB260924A.csv --model double \
        --break-epoch 1e5
        -> one break fixed at 1e5 s; the other break is free

    python fit_lightcurve.py data/GRB260924A.csv --model broken --common
        -> one broken power law shared by every band in the file

    python fit_lightcurve.py data/GRB260924A.csv --model broken --common \
        --bands g r z
        -> g, r, and z share the temporal indices and break time;
           every band in the file is drawn on one plot with that power law

    python fit_lightcurve.py data/GRB260924A.csv --model broken --common \
        --spectrum
        -> also writes the model spectrum at six log-spaced times from 1e3 s to 1e6 s
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.optimize import least_squares


# ============================================================
# Constants
# ============================================================

LN10 = np.log(10.0)

# Six observer times, equally spaced in log t from 1e3 s to 1e6 s.
SPECTRUM_EPOCHS_S = np.logspace(3, 6, 6)


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
# Detections in one band
# ============================================================


def band_log_data(df_band, break_epoch=None):
    """
    Positive detections for one band, converted to log10 flux.

    Upper limits are left out. Returns None when the band has no
    detections that can enter a fit.
    """

    detections = df_band[
        ~df_band["UL"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
    ].copy()

    detections = detections[
        (detections["Times"] > 0)
        & (detections["Fluxes"] > 0)
        & (detections["FluxErrs"] > 0)
    ].copy()

    if len(detections) == 0:
        return None

    t = detections["Times"].values.astype(float)
    f = detections["Fluxes"].values.astype(float)
    ferr = detections["FluxErrs"].values.astype(float)

    idx = np.argsort(t)

    t = t[idx]
    f = f[idx]
    ferr = ferr[idx]

    logt = np.log10(t)
    logf = np.log10(f)

    # sigma(log10 F) = sigma_F / (F ln 10)
    sigma_logf = np.maximum(ferr / (f * LN10), 1e-5)

    logtb_fixed = None if break_epoch is None else np.log10(break_epoch)

    return {
        "t": t,
        "f": f,
        "ferr": ferr,
        "logt": logt,
        "logf": logf,
        "sigma_logf": sigma_logf,
        "n": len(t),
        "logtb_fixed": logtb_fixed,
    }


# ============================================================
# Fit a single band
# ============================================================


def fit_band(df_band, model, break_epoch=None):

    obs = band_log_data(df_band, break_epoch)

    n = 0 if obs is None else obs["n"]

    nmin = minimum_points(model)

    if n < nmin:
        print(f"  Not enough detections for {model}: {n} < {nmin}")

        return None

    t = obs["t"]
    f = obs["f"]
    ferr = obs["ferr"]
    logt = obs["logt"]
    logf = obs["logf"]
    sigma_logf = obs["sigma_logf"]
    logtb_fixed = obs["logtb_fixed"]

    if break_epoch is not None and (break_epoch < t.min() or break_epoch > t.max()):
        print(
            f"  Fixed break at {break_epoch:.3g} s is outside the "
            f"detection range {t.min():.3g}–{t.max():.3g} s"
        )

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


def expand_shape(shape_free, model, logtb_fixed):
    """Shared temporal parameters, including any fixed break."""

    dummy = np.concatenate([[0.0], np.asarray(shape_free, dtype=float)])

    return expand_params(dummy, model, logtb_fixed)[1:]


def logF0_for_shape(logt, logf, shape, model):
    """Median normalization that places this band on a fixed shape."""

    baseline = evaluate_model(logt, np.concatenate([[0.0], shape]), model)

    return np.median(logf - baseline)


def common_residuals(free, prepared, model, logtb_fixed):

    n_bands = len(prepared)

    shape = expand_shape(free[n_bands:], model, logtb_fixed)

    if model == "double" and shape[4] <= shape[3]:
        n = sum(obs["n"] for _, obs in prepared)

        return np.ones(n) * 1e6

    pieces = []

    for logF0, (_, obs) in zip(free[:n_bands], prepared):
        params = np.concatenate([[logF0], shape])

        model_logf = evaluate_model(obs["logt"], params, model)

        pieces.append((obs["logf"] - model_logf) / obs["sigma_logf"])

    return np.concatenate(pieces)


def _band_fit_record(
    obs,
    params,
    errors,
    residuals_band,
    model,
    break_epoch,
    logtb_fixed,
    common,
    norm_only,
    joint,
):

    chi2_band = float(np.sum(residuals_band**2))

    record = {
        "params": params,
        "errors": errors,
        "t": obs["t"],
        "f": obs["f"],
        "ferr": obs["ferr"],
        "logt": obs["logt"],
        "logf": obs["logf"],
        "sigma_logf": obs["sigma_logf"],
        "residuals": residuals_band,
        "chi2": chi2_band,
        "n": obs["n"],
        "model": model,
        "break_epoch": break_epoch,
        "logtb_fixed": logtb_fixed,
        "common": common,
        "norm_only": norm_only,
    }

    if joint is not None and not norm_only:
        record["dof"] = joint["dof"]
        record["reduced_chi2"] = joint["reduced_chi2"]
        record["bic"] = joint["bic"]
        record["joint_chi2"] = joint["chi2"]
        record["joint_n"] = joint["n"]
        record["joint_k"] = joint["k"]

    else:
        k = 1
        dof = obs["n"] - k

        record["dof"] = dof
        record["reduced_chi2"] = chi2_band / dof if dof > 0 else np.nan
        record["bic"] = chi2_band + k * np.log(obs["n"])

    return record


def fit_normalization(obs, shape, model, break_epoch, logtb_fixed):
    """
    Place one band on a power law whose temporal shape is already fixed.
    """

    baseline = evaluate_model(obs["logt"], np.concatenate([[0.0], shape]), model)

    weight = 1.0 / obs["sigma_logf"] ** 2

    logF0 = float(np.sum(weight * (obs["logf"] - baseline)) / np.sum(weight))

    residuals_band = (obs["logf"] - baseline - logF0) / obs["sigma_logf"]

    chi2 = float(np.sum(residuals_band**2))

    dof = obs["n"] - 1

    scale = chi2 / dof if dof > 0 else 1.0

    err_logF0 = np.sqrt(scale / np.sum(weight))

    params = np.concatenate([[logF0], shape])

    errors = np.zeros(len(params))

    errors[0] = err_logF0

    return _band_fit_record(
        obs,
        params,
        errors,
        residuals_band,
        model,
        break_epoch,
        logtb_fixed,
        common=True,
        norm_only=True,
        joint=None,
    )


def fit_common(data, fit_bands, plot_bands, model, break_epoch=None):
    """
    One temporal power law for several bands.

    Temporal indices and break times are shared. Each band keeps its
    own logF0, because the bands do not have the same flux scale.

    Bands in plot_bands that did not enter the fit are drawn with that
    same shape; only their normalization is adjusted.
    """

    prepared = []
    no_detections = set()

    for band in fit_bands:
        df_band = data[data["Filt"].astype(str) == band]

        obs = band_log_data(df_band, break_epoch)

        if obs is None:
            print(f"  {band}: no detections, left out of the fit")

            no_detections.add(band)

            continue

        prepared.append((band, obs))

    n_total = sum(obs["n"] for _, obs in prepared)

    nmin = minimum_points(model)

    if n_total < nmin or len(prepared) == 0:
        print(f"  Not enough detections for a common {model} fit: {n_total} < {nmin}")

        return {}

    logtb_fixed = prepared[0][1]["logtb_fixed"]

    t_all = np.concatenate([obs["t"] for _, obs in prepared])

    if break_epoch is not None and (
        break_epoch < t_all.min() or break_epoch > t_all.max()
    ):
        print(
            f"  Fixed break at {break_epoch:.3g} s is outside the "
            f"detection range {t_all.min():.3g}–{t_all.max():.3g} s"
        )

    _, richest = max(prepared, key=lambda item: item[1]["n"])

    shape_free0 = initial_guess(
        richest["logt"], richest["logf"], model, logtb_fixed
    )[1:]

    shape0 = expand_shape(shape_free0, model, logtb_fixed)

    logF0s = np.array(
        [
            logF0_for_shape(obs["logt"], obs["logf"], shape0, model)
            for _, obs in prepared
        ]
    )

    logt_all = np.concatenate([obs["logt"] for _, obs in prepared])
    logf_all = np.concatenate([obs["logf"] for _, obs in prepared])

    lower_shape, upper_shape = parameter_bounds(
        logt_all, logf_all, model, logtb_fixed
    )

    lower_logF = []
    upper_logF = []

    for _, obs in prepared:
        lower_b, upper_b = parameter_bounds(
            obs["logt"], obs["logf"], model, logtb_fixed
        )

        lower_logF.append(lower_b[0])
        upper_logF.append(upper_b[0])

    lower = np.concatenate([lower_logF, lower_shape[1:]])
    upper = np.concatenate([upper_logF, upper_shape[1:]])

    dummy = np.concatenate([[0.0], shape_free0])

    dummy_lower = np.concatenate([[-100.0], lower_shape[1:]])
    dummy_upper = np.concatenate([[100.0], upper_shape[1:]])

    dummy = make_feasible_initial_guess(
        dummy, dummy_lower, dummy_upper, model, logtb_fixed
    )

    shape_free0 = dummy[1:]

    eps = 1e-8

    logF0s = np.clip(
        logF0s,
        lower[: len(logF0s)] + eps,
        upper[: len(logF0s)] - eps,
    )

    p0 = np.concatenate([logF0s, shape_free0])

    result = least_squares(
        common_residuals,
        p0,
        args=(prepared, model, logtb_fixed),
        bounds=(lower, upper),
        max_nfev=10000,
        loss="linear",
    )

    if not result.success:
        print(f"  Fit did not converge: {result.message}")

    best_free = result.x

    shape = expand_shape(best_free[len(prepared) :], model, logtb_fixed)

    res = common_residuals(best_free, prepared, model, logtb_fixed)

    chi2 = float(np.sum(res**2))

    k = len(best_free)

    dof = n_total - k

    reduced_chi2 = chi2 / dof if dof > 0 else np.nan

    bic = chi2 + k * np.log(n_total)

    try:
        # The break is a kink, so one Jacobian direction is numerical
        # noise. A pseudoinverse drops that direction and keeps finite
        # errors on the parameters the data actually constrain.
        covariance = np.linalg.pinv(result.jac.T @ result.jac, rcond=1e-10)

        if np.isfinite(reduced_chi2):
            covariance = covariance * reduced_chi2

        diag = np.diag(covariance)

        errors_free = np.sqrt(np.clip(diag, 0.0, None))

        errors_free[~np.isfinite(diag) | (diag < 0)] = np.nan

    except np.linalg.LinAlgError:
        errors_free = np.full(len(best_free), np.nan)

    if np.any(result.active_mask != 0):
        print("  A parameter finished on its bound.")

    joint = {
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "bic": bic,
        "n": n_total,
        "k": k,
    }

    fits = {}

    offset = 0

    for i, (band, obs) in enumerate(prepared):
        residuals_band = res[offset : offset + obs["n"]]

        offset += obs["n"]

        params = np.concatenate([[best_free[i]], shape])

        errors_band_free = np.concatenate(
            [[errors_free[i]], errors_free[len(prepared) :]]
        )

        errors = _embed_free_errors(errors_band_free, model, params, logtb_fixed)

        fits[band] = _band_fit_record(
            obs,
            params,
            errors,
            residuals_band,
            model,
            break_epoch,
            logtb_fixed,
            common=True,
            norm_only=False,
            joint=joint,
        )

    fitted = set(fits)

    for band in plot_bands:
        if band in fitted or band in no_detections:
            continue

        df_band = data[data["Filt"].astype(str) == band]

        obs = band_log_data(df_band, break_epoch)

        if obs is None:
            print(f"  {band}: no detections, data only")

            continue

        fits[band] = fit_normalization(
            obs, shape, model, break_epoch, logtb_fixed
        )

        print(f"  {band}: normalization only (not used in the fit)")

    return fits


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


def print_common_result(fits, fit_bands):
    """Shared temporal parameters, then one normalization per band."""

    fitted = [
        band
        for band in fit_bands
        if band in fits and not fits[band].get("norm_only")
    ]

    if not fitted:
        return

    proto = fits[fitted[0]]

    p = proto["params"]
    e = proto["errors"]

    print()
    print("=" * 70)
    print("Common power law: " + ", ".join(fitted))
    print("=" * 70)

    print(
        f"N = {proto['joint_n']}, "
        f"k = {proto['joint_k']}, "
        f"dof = {proto['dof']}, "
        f"chi2 = {proto['joint_chi2']:.3f}, "
        f"reduced chi2 = {proto['reduced_chi2']:.3f}, "
        f"BIC = {proto['bic']:.3f}"
    )

    if proto["model"] == "single":
        print(f"alpha = {p[1]:.5f} +/- {e[1]:.5f}")

    elif proto["model"] == "broken":
        print(f"alpha1 = {p[1]:.5f} +/- {e[1]:.5f}")

        print(f"alpha2 = {p[2]:.5f} +/- {e[2]:.5f}")

        tb_fixed = proto.get("logtb_fixed") is not None

        print(f"log10(tb/s) = {_format_value(p[3], e[3], tb_fixed)}")

        print(f"tb = {10 ** p[3]:.3g} s")

    elif proto["model"] == "double":
        print(f"alpha1 = {p[1]:.5f} +/- {e[1]:.5f}")

        print(f"alpha2 = {p[2]:.5f} +/- {e[2]:.5f}")

        print(f"alpha3 = {p[3]:.5f} +/- {e[3]:.5f}")

        fixed_at = None

        if proto.get("logtb_fixed") is not None:
            fixed_at = fixed_break_index(p, proto["logtb_fixed"])

        print(f"log10(tb1/s) = {_format_value(p[4], e[4], fixed_at == 4)}")

        print(f"log10(tb2/s) = {_format_value(p[5], e[5], fixed_at == 5)}")

        print(f"tb1 = {10 ** p[4]:.3g} s")

        print(f"tb2 = {10 ** p[5]:.3g} s")

    print()
    print("Normalizations")

    for band in fitted:
        fit = fits[band]

        print(
            f"  {band}: logF0 = {fit['params'][0]:.5f} +/- {fit['errors'][0]:.5f}"
            f"    N = {fit['n']}, chi2 = {fit['chi2']:.3f}"
        )

    norm_only = [
        band for band, fit in fits.items() if fit.get("norm_only") and band not in fitted
    ]

    if norm_only:
        print()
        print("Normalization only (shape fixed from the bands above)")

        for band in norm_only:
            fit = fits[band]

            print(
                f"  {band}: logF0 = {fit['params'][0]:.5f} +/- {fit['errors'][0]:.5f}"
                f"    N = {fit['n']}, chi2 = {fit['chi2']:.3f}"
            )


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


def label_powerlaw_slopes(ax, tmin, tmax, params, model, color, flux_range=None):
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

        # Keep the label on the part of the curve that is inside the axes.
        if flux_range is not None and not (flux_range[0] < f_mid < flux_range[1]):
            samples = np.logspace(np.log10(t_left), np.log10(t_right), 40)
            fluxes = 10 ** evaluate_model(np.log10(samples), params, model)
            inside = (fluxes > flux_range[0]) & (fluxes < flux_range[1])

            if not np.any(inside):
                continue

            nearest = np.argmin(np.abs(np.log10(samples[inside]) - np.log10(t_mid)))
            t_mid = samples[inside][nearest]
            f_mid = fluxes[inside][nearest]

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


def _is_upper_limit(series):

    return series.astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])


def _band_colors(bands):
    """Distinct color for each filter, stable for a given band list."""

    cmap = plt.get_cmap("tab20")

    return {band: cmap(i % cmap.N) for i, band in enumerate(bands)}


def plot_common_results(data, fits, model, bands, output=None):
    """
    All bands on one light curve, with one residual panel underneath.

    A band that entered the fit is a solid curve. A band whose
    normalization was set afterwards is dashed.
    """

    fig, (ax, axres) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    plotted = data[data["Filt"].astype(str).isin(bands)]
    times = plotted.loc[plotted["Times"] > 0, "Times"]
    t_lo = float(times.min())
    t_hi = float(times.max())

    fluxes = plotted.loc[
        (plotted["Times"] > 0) & (plotted["Fluxes"] > 0),
        "Fluxes",
    ]
    f_lo = float(fluxes.min())
    f_hi = float(fluxes.max())

    colors = _band_colors(bands)
    legend_handles = []
    residual_sets = []
    reference = None

    tmodel = np.logspace(np.log10(t_lo), np.log10(t_hi), 500)
    logtmodel = np.log10(tmodel)

    for band in bands:
        color = colors[band]
        df_band = data[data["Filt"].astype(str) == band]
        fit = fits.get(band)

        detections = df_band[
            ~_is_upper_limit(df_band["UL"])
            & (df_band["Times"] > 0)
            & (df_band["Fluxes"] > 0)
        ]

        upper = df_band[
            _is_upper_limit(df_band["UL"])
            & (df_band["Times"] > 0)
            & (df_band["Fluxes"] > 0)
        ]

        if len(detections) > 0:
            ax.errorbar(
                detections["Times"],
                detections["Fluxes"],
                yerr=detections["FluxErrs"],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                markersize=5,
                linestyle="none",
                zorder=3,
            )

        if len(upper) > 0:
            ax.errorbar(
                upper["Times"],
                upper["Fluxes"],
                yerr=0.15 * upper["Fluxes"],
                uplims=True,
                fmt="v",
                color=color,
                ecolor=color,
                markersize=6,
                linestyle="none",
                zorder=3,
            )

        linestyle = "none"
        label = band

        if fit is not None:
            fmodel = 10 ** evaluate_model(logtmodel, fit["params"], model)

            if fit.get("norm_only"):
                linestyle = "--"
                label = f"{band} (norm.)"

            else:
                linestyle = "-"

            ax.plot(
                tmodel,
                fmodel,
                linestyle=linestyle,
                color=color,
                linewidth=1.8,
                zorder=2,
            )

            axres.errorbar(
                fit["t"],
                fit["residuals"],
                fmt="o",
                color=color,
                markersize=4,
                linestyle="none",
                zorder=3,
            )

            residual_sets.append(fit["residuals"])

            span = np.log10(fit["t"].max()) - np.log10(fit["t"].min())
            rank = (0 if fit.get("norm_only") else 1, span)

            if reference is None or rank > reference[0]:
                reference = (rank, fit)

        elif len(upper) > 0 and len(detections) == 0:
            label = f"{band} (limit)"
            linestyle = "none"

        elif len(detections) == 0 and len(upper) == 0:
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="v" if label.endswith("(limit)") else "o",
                linestyle=linestyle,
                markersize=5,
                label=label,
            )
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    axres.set_xscale("log")

    ax.set_xlim(t_lo, t_hi)

    if np.isclose(f_lo, f_hi):
        ax.set_ylim(f_lo / 3.0, f_hi * 3.0)

    else:
        pad = 0.12 * (np.log10(f_hi) - np.log10(f_lo))

        ax.set_ylim(
            10 ** (np.log10(f_lo) - pad),
            10 ** (np.log10(f_hi) + pad),
        )

    ax.set_ylabel("Flux density (mJy)")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8,
        frameon=True,
    )

    axres.axhline(0, color="0.4", linestyle="--", linewidth=1, zorder=1)
    axres.set_ylabel(r"Residual ($\sigma$)")
    axres.set_xlabel("Time since trigger (s)")
    axres.grid(True, which="both", alpha=0.3)

    if residual_sets:
        residual_sigma = np.concatenate(residual_sets)

        axres.set_ylim(
            min(-4, np.min(residual_sigma) - 1),
            max(4, np.max(residual_sigma) + 1),
        )

    else:
        axres.set_ylim(-4, 4)

    if reference is not None:
        fit = reference[1]

        label_powerlaw_slopes(
            ax,
            t_lo,
            t_hi,
            fit["params"],
            model,
            "black",
            flux_range=ax.get_ylim(),
        )

        text = (
            rf"joint $\chi^2_\nu={fit['reduced_chi2']:.2f}$"
            "\n"
            rf"BIC = {fit['bic']:.1f}"
        )

        if fit.get("norm_only"):
            text = (
                "normalization only"
                "\n"
                rf"$\chi^2={fit['chi2']:.2f}$"
            )

        ax.text(
            0.03,
            0.05,
            text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", alpha=0.8),
            zorder=6,
        )

        p = fit["params"]

        if model == "broken":
            mark_break(ax, axres, 10 ** p[3])

        elif model == "double":
            mark_break(ax, axres, 10 ** p[4])
            mark_break(ax, axres, 10 ** p[5])

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight", format="pdf")

        print(f"\nSaved figure to: {output}")

    else:
        plt.show()

    plt.close(fig)


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
# Spectrum from the light-curve fit
# ============================================================


def band_frequency(data, band):
    """Median observing frequency (Hz) of one filter. None if unknown."""

    nu = pd.to_numeric(
        data.loc[data["Filt"].astype(str) == band, "Freqs"],
        errors="coerce",
    )

    nu = nu[np.isfinite(nu) & (nu > 0)]

    if len(nu) == 0:
        return None

    return float(np.median(nu))


def spectrum_from_fit(data, fits, epochs=None):
    """
    Model flux density at each fitted band, at fixed observer times.

    The spectrum runs from the lowest to the highest frequency among
    bands that have both a frequency and a light-curve fit. Epochs
    outside the time span of the fitted detections are still evaluated;
    those rows are marked extrapolated.

    Returns a table with epoch_s, band, frequency_Hz, flux_mJy,
    extrapolated. None when nothing can be evaluated.
    """

    if epochs is None:
        epochs = SPECTRUM_EPOCHS_S

    epochs = np.asarray(epochs, dtype=float)

    usable = []

    for band, fit in fits.items():
        if not fit:
            continue

        nu = band_frequency(data, band)

        if nu is None:
            print(f"  {band}: no frequency, left out of the spectrum")

            continue

        usable.append((band, nu, fit))

    if not usable:
        return None

    t_fit = np.concatenate([fit["t"] for _, _, fit in usable])
    t_lo = float(t_fit.min())
    t_hi = float(t_fit.max())

    rows = []

    for epoch in epochs:
        extrapolated = bool(epoch < t_lo or epoch > t_hi)
        logt = np.log10(epoch)

        for band, nu, fit in usable:
            logf = evaluate_model(np.array([logt]), fit["params"], fit["model"])[0]

            rows.append(
                {
                    "epoch_s": float(epoch),
                    "band": band,
                    "frequency_Hz": nu,
                    "flux_mJy": float(10 ** logf),
                    "extrapolated": extrapolated,
                }
            )

    spectrum = pd.DataFrame(rows)

    spectrum = spectrum.sort_values(
        ["epoch_s", "frequency_Hz", "band"], kind="mergesort"
    )

    return spectrum.reset_index(drop=True)


def _spectrum_curve(frequency, flux):
    """Frequency-sorted curve, averaging bands that share a frequency."""

    order = np.argsort(frequency)
    frequency = np.asarray(frequency, dtype=float)[order]
    flux = np.asarray(flux, dtype=float)[order]

    nu_out = []
    flux_out = []
    i = 0

    while i < len(frequency):
        j = i + 1

        while j < len(frequency) and np.isclose(frequency[j], frequency[i], rtol=1e-6):
            j += 1

        nu_out.append(frequency[i])
        flux_out.append(float(np.exp(np.mean(np.log(flux[i:j])))))
        i = j

    return np.array(nu_out), np.array(flux_out)


def spectrum_table(spectrum):
    """
    One row per band. Each epoch is its own flux column, low frequency first.
    """

    base = (
        spectrum[["band", "frequency_Hz"]]
        .drop_duplicates()
        .sort_values(["frequency_Hz", "band"], kind="mergesort")
        .set_index(["band", "frequency_Hz"])
    )

    for epoch, piece in spectrum.groupby("epoch_s", sort=True):
        column = f"flux_mJy_{epoch:.0f}s"
        values = piece.set_index(["band", "frequency_Hz"])["flux_mJy"]
        base[column] = values

    return base.reset_index()


def plot_model_spectrum(spectrum, output=None):
    """One plot: every epoch is a different color."""

    epochs = list(dict.fromkeys(spectrum["epoch_s"].tolist()))
    colors = [f"C{i % 10}" for i in range(len(epochs))]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for epoch, color in zip(epochs, colors):
        piece = spectrum[np.isclose(spectrum["epoch_s"], epoch)]
        nu, flux = _spectrum_curve(piece["frequency_Hz"], piece["flux_mJy"])

        extrapolated = bool(piece["extrapolated"].iloc[0])
        label = f"{epoch:.3g} s"

        if extrapolated:
            label = label + " (extrap.)"

        ax.plot(
            nu,
            flux,
            "-o",
            color=color,
            linewidth=1.8,
            markersize=5,
            label=label,
            zorder=3,
        )

    nu_all = spectrum["frequency_Hz"].to_numpy()
    flux_all = spectrum["flux_mJy"].to_numpy()

    log_nu = np.log10(nu_all)
    pad_nu = 0.04 * max(log_nu.max() - log_nu.min(), 0.3)

    ax.set_xlim(10 ** (log_nu.min() - pad_nu), 10 ** (log_nu.max() + pad_nu))

    log_f = np.log10(flux_all)
    pad_f = 0.08 * max(log_f.max() - log_f.min(), 0.3)

    ax.set_ylim(10 ** (log_f.min() - pad_f), 10 ** (log_f.max() + pad_f))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"Flux density (mJy)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight", format="pdf")

        print(f"Saved spectrum to: {output}")

    else:
        plt.show()

    plt.close(fig)


def print_spectrum(spectrum):
    """Print the model spectrum, one epoch at a time."""

    nu_lo = spectrum["frequency_Hz"].min()
    nu_hi = spectrum["frequency_Hz"].max()

    print()
    print("=" * 70)
    print(f"Model spectrum: {nu_lo:.3g}–{nu_hi:.3g} Hz")
    print("=" * 70)

    for epoch, piece in spectrum.groupby("epoch_s", sort=True):
        note = " (extrapolated)" if bool(piece["extrapolated"].iloc[0]) else ""

        print()
        print(f"{epoch:.6g} s{note}")

        for row in piece.itertuples(index=False):
            print(
                f"  {row.band:<16s}  {row.frequency_Hz:.6g} Hz"
                f"    {row.flux_mJy:.6g} mJy"
            )


def write_spectrum(data, fits, lightcurve_output):
    """Evaluate, print, and save the spectrum next to the light-curve figure."""

    print()
    print("Computing the model spectrum")

    spectrum = spectrum_from_fit(data, fits)

    if spectrum is None or len(spectrum) == 0:
        print("  No fitted band has a frequency; spectrum not written.")

        return None

    print_spectrum(spectrum)

    path = Path(lightcurve_output)
    csv_path = path.with_name(path.stem + "_spectrum.csv")
    pdf_path = path.with_name(path.stem + "_spectrum.pdf")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum_table(spectrum).to_csv(csv_path, index=False)

    print(f"\nSaved spectrum table to: {csv_path}")

    plot_model_spectrum(spectrum, str(pdf_path))

    return spectrum


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
        help=(
            "Filters to fit. If omitted, every filter in the file is fit. "
            "With --common, these bands share one power law."
        ),
    )

    parser.add_argument(
        "--common",
        action="store_true",
        help=(
            "Fit one power law to the selected bands at once. "
            "Temporal indices and break times are shared; each band "
            "has its own normalization. Every filter in the file is "
            "drawn together on one plot."
        ),
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
        "--spectrum",
        action="store_true",
        help=(
            "Evaluate the fitted flux from the lowest to the highest "
            "frequency in the data at six times spaced logarithmically "
            "from 1e3 s to 1e6 s, and write a spectrum plot and table."
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

    in_file = sorted(data["Filt"].astype(str).unique())

    if args.bands is None:
        fit_bands = in_file

    else:
        unknown = [band for band in args.bands if band not in set(in_file)]

        if unknown:
            print("Bands not in the file (skipped): " + ", ".join(unknown))

        fit_bands = [band for band in args.bands if band in set(in_file)]

    if args.common:
        plot_bands = list(
            dict.fromkeys(
                fit_bands + [band for band in in_file if band not in set(fit_bands)]
            )
        )

    else:
        plot_bands = fit_bands

    print()
    print("=" * 70)
    print("GRB LIGHT-CURVE FIT")
    print("=" * 70)

    print(f"Input : {args.input}")

    print(f"Model : {args.model}")

    if args.common:
        print("Mode  : common power law")

        print(f"Fit   : {', '.join(fit_bands) if fit_bands else '(none)'}")

        print(f"Plot  : {', '.join(plot_bands) if plot_bands else '(none)'}")

    else:
        print(f"Bands : {', '.join(plot_bands) if plot_bands else '(none)'}")

    if args.break_epoch is not None:
        print(f"Break : {args.break_epoch:.6g} s (fixed)")

    if args.output is None:
        stem = Path(args.input).stem
        band_part = "_".join(fit_bands) if fit_bands else "none"
        model_part = f"common_{args.model}" if args.common else args.model
        output = Path("output") / f"{stem}_{band_part}_{model_part}.pdf"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output = str(output)

    print(f"Output: {output}")

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fits = {}

    if args.common:
        print()
        print(
            "Fitting a common "
            f"{args.model} power law to: {', '.join(fit_bands)}"
        )

        fits = fit_common(
            data, fit_bands, plot_bands, args.model, args.break_epoch
        )

        print_common_result(fits, fit_bands)

    else:
        for band in fit_bands:
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

    if plot_bands and args.common:
        plot_common_results(data, fits, args.model, plot_bands, output)

    elif plot_bands:
        plot_results(data, fits, args.model, plot_bands, output)

    else:
        print("\nNo bands to plot; figure not written.")

    if args.spectrum:
        write_spectrum(data, fits, output)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
