#!/usr/bin/env python3
"""Rest-frame Fourier comparison of the five most energetic Fermi GRBs.

GRB 260924A is included with them. Its isotropic energy is the rest-frame
10 keV–10 MeV value measured for this burst, which places it second.

The sample is the five bursts in ``high_energy_grb.csv`` that Fermi detected
and that have the largest isotropic energies. Light curves are Fermi-GBM
CTIME, NaI detectors only, in the channels that lie fully inside 8–1000 keV.
The three detectors with the highest signal-to-noise are summed, skipping a
detector whose livetime collapses during the peak.

The power density spectrum is the Leahy periodogram of the background-
subtracted light curve. The white-noise level is 2 and is removed before the
fit. Frequencies are shifted to the source frame by (1 + z). The comparison
index is a single power law fit over the rest-frame band that every burst
actually measures.
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fourier_gbm"
OUT = ROOT / "results" / "fourier_gbm"
BASE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts"
EMIN_KEV = 8.0
EMAX_KEV = 1000.0
N_DETECTORS = 3
LIVE_FRAC_MIN = 0.50
BINS_PER_DECADE = 8

# Eiso is 10^52 erg, taken from high_energy_grb.csv. Redshifts are the same file.
BURSTS = [
    {"name": "221009A", "trigger": "bn221009553", "z": 0.1510, "eiso": 1500.00, "t90_cat": 600.00},
    {"name": "260924A", "trigger": "bn260924051", "z": 3.51, "eiso": 995.19, "t90_cat": 337.93},
    {"name": "190530A", "trigger": "bn190530430", "z": 0.9386, "eiso": 605.00, "t90_cat": 18.43},
    {"name": "090323", "trigger": "bn090323002", "z": 3.5832, "eiso": 581.00, "t90_cat": 135.17},
    {"name": "160625B", "trigger": "bn160625945", "z": 1.4060, "eiso": 510.10, "t90_cat": 454.67},
    {"name": "080916C", "trigger": "bn080916009", "z": 4.3500, "eiso": 371.24, "t90_cat": 62.98},
]
COLORS = {
    "221009A": "#000000",
    "260924A": "#882255",
    "190530A": "#D55E00",
    "090323": "#0072B2",
    "160625B": "#009E73",
    "080916C": "#CC79A7",
}


def _request(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "grb-fourier-comparison"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _listing(url):
    html = _request(url, timeout=90).decode("utf-8", "replace")
    return re.findall(r'href="([^"?#/][^"?#]*)"', html)


def _latest(names):
    best = {}
    for name in names:
        match = re.search(r"_v(\d+)\.", name)
        if match is None:
            continue
        version = int(match.group(1))
        key = name.split("_")[2]
        if key not in best or version > best[key][0]:
            best[key] = (version, name)
    return [name for _, name in best.values()]


def download_burst(burst):
    year = "20" + burst["trigger"][2:4]
    page = f"{BASE}/{year}/{burst['trigger']}/current/"
    names = [
        name for name in _listing(page)
        if re.fullmatch(rf"glg_ctime_n[0-9ab]_{re.escape(burst['trigger'])}_v\d+\.pha", name)
    ]
    if not names:
        raise FileNotFoundError(f"No NaI CTIME files at {page}")
    dest_dir = DATA / burst["trigger"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(dest_dir.glob(f"glg_ctime_n*_{burst['trigger']}_v*.pha"))
    if len(cached) >= 12:
        print(f"  {burst['name']}: using {len(cached)} CTIME files already on disk")
        return cached
    elsewhere = sorted((ROOT / "data" / f"GRB{burst['name']}" / "gbm").glob(
        f"glg_ctime_n*_{burst['trigger']}_v*.pha"
    ))
    if len(elsewhere) >= 12:
        print(f"  {burst['name']}: using {len(elsewhere)} CTIME files from data/GRB{burst['name']}/gbm")
        return elsewhere
    paths = []
    for name in _latest(names):
        dest = dest_dir / name
        paths.append(dest)
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        body = _request(page + name)
        if not body.startswith(b"SIMPLE"):
            raise RuntimeError(f"{name} is not a FITS file ({len(body)} bytes)")
        partial = dest.with_suffix(dest.suffix + ".part")
        partial.write_bytes(body)
        partial.replace(dest)
        print(f"  {burst['name']}: {name}  {len(body) / 1e6:.1f} MB")
    return paths


def read_detector(path):
    """Native CTIME rows for the channels lying fully inside 8–1000 keV."""
    with fits.open(path) as hdus:
        bounds = hdus["EBOUNDS"].data
        spectrum = hdus["SPECTRUM"].data
        header = hdus["SPECTRUM"].header
        trigger = float(header["TRIGTIME"])
        keep = (bounds["E_MIN"] >= EMIN_KEV - 1e-6) & (bounds["E_MAX"] <= EMAX_KEV + 1e-6)
        if not np.any(keep):
            keep = (bounds["E_MAX"] > EMIN_KEV) & (bounds["E_MIN"] < EMAX_KEV)
        energy = (float(bounds["E_MIN"][keep].min()), float(bounds["E_MAX"][keep].max()))
        t0 = np.asarray(spectrum["TIME"], float) - trigger
        t1 = np.asarray(spectrum["ENDTIME"], float) - trigger
        counts = np.asarray(spectrum["COUNTS"], float)[:, keep].sum(axis=1)
        exposure = np.asarray(spectrum["EXPOSURE"], float)
    order = np.argsort(t0)
    detector = path.name.split("_")[2]
    return {
        "detector": detector,
        "t0": t0[order],
        "t1": t1[order],
        "counts": counts[order],
        "exposure": exposure[order],
        "energy": energy,
    }


def high_resolution_slice(detector):
    """Contiguous 64 ms segment. CTIME is 256 ms outside the triggered interval."""
    width = np.round(detector["t1"] - detector["t0"], 4)
    fast = np.abs(width - 0.064) < 0.005
    if fast.sum() < 50:
        fast = np.abs(width - 0.256) < 0.02
        native = 0.256
    else:
        native = 0.064
    if fast.sum() < 50:
        raise RuntimeError(f"{detector['detector']}: no uniform CTIME segment")
    indices = np.flatnonzero(fast)
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    stops = np.r_[indices[breaks], indices[-1]]
    # The triggered segment is the one that contains the trigger, else the longest.
    chosen = None
    for start, stop in zip(starts, stops):
        if detector["t0"][start] <= 0.0 <= detector["t1"][stop]:
            chosen = (start, stop)
            break
    if chosen is None:
        lengths = stops - starts
        start = int(starts[np.argmax(lengths)])
        chosen = (start, int(stops[np.argmax(lengths)]))
    start, stop = chosen
    return start, stop + 1, native


def _poly_rate(time, rate, weight, order, evaluate):
    if order == 0 or len(time) < order + 3:
        return np.full_like(evaluate, np.average(rate, weights=weight), dtype=float)
    coefficient = np.polyfit(time, rate, order, w=weight)
    return np.polyval(coefficient, evaluate)


def fit_background(detector, start, stop):
    """Polynomial rate from pre-trigger bins, plus a post-burst anchor when it is quiet."""
    t0, t1 = detector["t0"], detector["t1"]
    counts, exposure = detector["counts"], detector["exposure"]
    mid = 0.5 * (t0 + t1)
    width = t1 - t0
    usable = (exposure > 0.5 * width) & (counts >= 0)
    before = usable & (mid < -20.0) & (mid > -250.0)
    if before.sum() < 20:
        before = usable & (mid < t0[start] - 5.0) & (mid > t0[start] - 400.0)
    if before.sum() < 10:
        raise RuntimeError(f"{detector['detector']}: not enough pre-trigger background")
    emission_end = float(t1[stop - 1])
    after = usable & (mid > emission_end + 30.0) & (mid < emission_end + 250.0)
    pre_rate = counts[before] / exposure[before]
    pre_level = float(np.median(pre_rate))
    if after.sum() >= 15:
        post_level = float(np.median(counts[after] / exposure[after]))
        if not (0.5 * pre_level <= post_level <= 2.0 * pre_level):
            after = np.zeros_like(after)
    use = before | after
    # One clip against a constant model, so a data glitch does not tilt the polynomial.
    rate = counts[use] / exposure[use]
    deviation = np.abs(rate - np.median(rate))
    scale = np.median(deviation) * 1.4826
    if scale > 0:
        use_index = np.flatnonzero(use)
        use_index = use_index[deviation < 5.0 * scale]
    else:
        use_index = np.flatnonzero(use)
    time = mid[use_index]
    rate = counts[use_index] / exposure[use_index]
    weight = np.sqrt(np.maximum(exposure[use_index], 1e-3))
    evaluate = mid[start:stop]
    order = 1 if (after.any() and len(time) >= 40) else 0
    model = _poly_rate(time, rate, weight, order, evaluate)
    if np.any(model <= 0) or np.max(model) > 3.0 * pre_level or np.min(model) < 0.4 * pre_level:
        order = 0
        model = _poly_rate(time, rate, weight, order, evaluate)
    return model, order, pre_level


def select_and_sum(detectors):
    """Sum the three highest-significance NaIs that stay live through the peak."""
    prepared = []
    for detector in detectors:
        try:
            start, stop, native = high_resolution_slice(detector)
            background, order, _ = fit_background(detector, start, stop)
        except RuntimeError as exc:
            print(f"  skipping {detector['detector']}: {exc}")
            continue
        counts = detector["counts"][start:stop]
        exposure = detector["exposure"][start:stop]
        mid = 0.5 * (detector["t0"][start:stop] + detector["t1"][start:stop])
        rate = np.divide(counts, exposure, out=np.zeros_like(counts), where=exposure > 0)
        net_rate = rate - background
        # Rank on a fixed 0–min(600 s, coverage) window so a later tail cannot hide the peak.
        rank = (mid >= -5.0) & (mid <= min(600.0, float(mid[-1])))
        net = float(np.sum(net_rate[rank] * exposure[rank]))
        variance = float(np.sum(counts[rank]))
        significance = net / np.sqrt(variance) if variance > 0 else -np.inf
        peak = net_rate > np.quantile(net_rate, 0.99)
        live = float(np.median(exposure[peak] / native))
        prepared.append({
            "detector": detector["detector"],
            "mid": mid,
            "net_rate": net_rate,
            "var_rate": np.divide(counts, exposure ** 2, out=np.zeros_like(counts), where=exposure > 0),
            "rate": rate,
            "background": background,
            "significance": significance,
            "live_fraction": live,
            "order": order,
            "native": native,
            "energy": detector["energy"],
            "saturated": live < LIVE_FRAC_MIN,
        })
    prepared.sort(key=lambda row: row["significance"], reverse=True)
    # Prefer detectors that stay above 70% livetime at the peak. GRB 221009A
    # drives every NaI into deadtime; the least-affected ones distort the
    # fast variability less than the ones with the most counts.
    healthy = [row for row in prepared if row["live_fraction"] >= 0.70 and row["significance"] > 0]
    deadtime = False
    if len(healthy) >= 2:
        chosen = healthy[:N_DETECTORS]
    else:
        deadtime = True
        usable = [row for row in prepared if not row["saturated"] and row["significance"] > 0]
        chosen = (usable or prepared)[:N_DETECTORS]
    if len(chosen) < 2:
        raise RuntimeError("Fewer than two usable NaI detectors")
    native = chosen[0]["native"]
    if any(row["native"] != native for row in chosen):
        raise RuntimeError("Selected detectors are not on the same time grid")
    reference = chosen[0]["mid"]
    for row in chosen[1:]:
        if len(row["mid"]) != len(reference) or np.max(np.abs(row["mid"] - reference)) > 1e-3:
            raise RuntimeError(f"{row['detector']} is not aligned with {chosen[0]['detector']}")
    summed = {
        "mid": reference,
        "dt": native,
        "net_rate": np.sum([row["net_rate"] for row in chosen], axis=0),
        "var_rate": np.sum([row["var_rate"] for row in chosen], axis=0),
        "rate": np.sum([row["rate"] for row in chosen], axis=0),
        "background": np.sum([row["background"] for row in chosen], axis=0),
        "detectors": [row["detector"] for row in chosen],
        "rejected_saturated": [row["detector"] for row in prepared if row["saturated"]],
        "live_fraction": min(row["live_fraction"] for row in chosen),
        "deadtime": deadtime,
        "background_order": max(row["order"] for row in chosen),
        "energy": chosen[0]["energy"],
        "ranking": [(row["detector"], row["significance"], row["live_fraction"]) for row in prepared],
    }
    return summed


def emission_window(summed):
    """Interval holding 5–95% of the net counts, extended to the pulse edges.

    A noise spike late in the CTIME segment must not stretch the window.
    The cumulative fluence sets the core, and the edges walk outward only
    while the 1 s net rate stays above the local noise.
    """
    dt = summed["dt"]
    net = summed["net_rate"]
    noise = np.sqrt(np.maximum(summed["var_rate"], 0.0))
    box = max(1, int(round(1.0 / dt)))
    kernel = np.ones(box) / box
    smooth = np.convolve(net, kernel, mode="same")
    smooth_noise = np.convolve(noise, kernel, mode="same") / np.sqrt(box)
    smooth_noise = np.maximum(smooth_noise, 1e-6)
    # Positive noise integrates to a fake tail and pushes the 95% point
    # hundreds of seconds past the burst. Only bins that clear 5 sigma count.
    significant = smooth > 5.0 * smooth_noise
    fluence = np.cumsum(np.where(significant, np.clip(net, 0.0, None), 0.0)) * dt
    total = float(fluence[-1])
    if total <= 0:
        raise RuntimeError("No net fluence")
    core_start = int(np.searchsorted(fluence, 0.05 * total))
    core_stop = int(np.searchsorted(fluence, 0.95 * total))
    above = smooth > 3.0 * smooth_noise
    start = core_start
    while start > 0 and above[start]:
        start -= 1
    stop = min(core_stop, len(net) - 1)
    while stop < len(net) - 1 and above[stop]:
        stop += 1
    pad = box
    start = max(0, start - pad)
    stop = min(len(net), stop + pad + 1)
    if stop - start < 4 * box:
        raise RuntimeError("Emission window is shorter than 4 s")
    return start, stop


def periodogram(summed, start, stop):
    dt = summed["dt"]
    net = summed["net_rate"][start:stop]
    var_rate = summed["var_rate"][start:stop]
    if np.any(~np.isfinite(net)) or np.any(var_rate < 0):
        raise RuntimeError("Non-finite light curve")
    counts_like = net * dt
    variance = var_rate * dt ** 2
    centered = counts_like - np.mean(counts_like)
    transform = np.fft.rfft(centered)
    sum_var = float(np.sum(variance))
    if sum_var <= 0:
        raise RuntimeError("Zero variance")
    # Drop the mean bin and the Nyquist bin. Leahy power of Poisson noise is 2.
    power = 2.0 * np.abs(transform[1:-1]) ** 2 / sum_var
    frequency = np.fft.rfftfreq(len(centered), dt)[1:-1]
    return frequency, power, sum_var


def log_rebin(frequency, power, bins_per_decade=BINS_PER_DECADE, min_modes=3):
    if len(frequency) < min_modes:
        raise RuntimeError("Periodogram is too short to rebin")
    log_start = np.log10(frequency[0])
    log_stop = np.log10(frequency[-1])
    edges = 10 ** np.arange(log_start, log_stop, 1.0 / bins_per_decade)
    if edges[-1] < frequency[-1]:
        edges = np.r_[edges, frequency[-1]]
    centres, means, errors, counts, lows, highs = [], [], [], [], [], []
    left = 0
    for right_edge in edges[1:]:
        right = int(np.searchsorted(frequency, right_edge, side="right"))
        if right - left < min_modes:
            continue
        chunk = power[left:right]
        mean = float(np.mean(chunk))
        centres.append(float(np.exp(np.mean(np.log(frequency[left:right])))))
        means.append(mean)
        errors.append(mean / np.sqrt(right - left))
        counts.append(right - left)
        lows.append(float(frequency[left]))
        highs.append(float(frequency[right - 1]))
        left = right
    if not centres:
        raise RuntimeError("Log rebin produced no bins")
    return {
        "f": np.asarray(centres),
        "f_lo": np.asarray(lows),
        "f_hi": np.asarray(highs),
        "P": np.asarray(means),
        "P_err": np.asarray(errors),
        "n": np.asarray(counts),
    }


def _powerlaw(log_f, norm, alpha, log_f0):
    return norm - alpha * (log_f - log_f0)


def fit_powerlaw(frequency, power_net, power_err):
    """Weighted fit of log10 P = c - alpha log10(f / f0). Returns alpha and the model."""
    usable = (power_net > 0) & (power_err > 0) & np.isfinite(power_net)
    frequency, power_net, power_err = frequency[usable], power_net[usable], power_err[usable]
    if len(frequency) < 4:
        return None
    log_f = np.log10(frequency)
    log_p = np.log10(power_net)
    sigma = power_err / (power_net * np.log(10.0))
    log_f0 = float(np.average(log_f, weights=1.0 / sigma ** 2))
    slope, intercept = np.polyfit(log_f - log_f0, log_p, 1, w=1.0 / sigma)
    alpha = float(-slope)
    # Covariance from a linear weighted least-squares fit.
    design = np.column_stack([log_f - log_f0, np.ones_like(log_f)])
    try:
        covariance = np.linalg.inv(design.T @ (design / sigma[:, None] ** 2))
        alpha_err = float(np.sqrt(covariance[0, 0]))
    except np.linalg.LinAlgError:
        alpha_err = np.nan
    model = 10 ** (intercept + slope * (log_f - log_f0))
    chi2 = float(np.sum(((log_p - np.log10(model)) / sigma) ** 2))
    return {
        "alpha": alpha,
        "alpha_err": alpha_err,
        "f0": 10 ** log_f0,
        "f_min": float(frequency[0]),
        "f_max": float(frequency[-1]),
        "chi2": chi2,
        "dof": int(len(frequency) - 2),
        "n": int(len(frequency)),
    }


def _broken(log_f, log_a, alpha1, alpha2, log_fb):
    f_ratio = 10 ** (log_f - log_fb)
    sharpness = 2.0
    bending = (alpha1 - alpha2) / sharpness * np.log10(1.0 + f_ratio ** sharpness)
    return log_a - alpha1 * (log_f - log_fb) + bending


def fit_broken(frequency, power_net, power_err):
    usable = (power_net > 0) & (power_err > 0) & np.isfinite(power_net)
    frequency, power_net, power_err = frequency[usable], power_net[usable], power_err[usable]
    if len(frequency) < 8 or frequency[-1] / frequency[0] < 8:
        return None
    log_f = np.log10(frequency)
    log_p = np.log10(power_net)
    sigma = power_err / (power_net * np.log(10.0))
    log_fb0 = float(np.median(log_f))
    guess = [float(np.median(log_p)), 1.0, 1.8, log_fb0]
    lower = [-2.0, 0.0, 0.2, log_f[0] + 0.15]
    upper = [8.0, 3.5, 4.5, log_f[-1] - 0.15]
    try:
        fitted, covariance = curve_fit(
            _broken, log_f, log_p, p0=guess, sigma=sigma, absolute_sigma=True,
            bounds=(lower, upper), maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    model = _broken(log_f, *fitted)
    chi2 = float(np.sum(((log_p - model) / sigma) ** 2))
    errors = np.sqrt(np.diag(covariance))
    fb = 10 ** float(fitted[3])
    interior = frequency[1] * 1.5 < fb < frequency[-2] / 1.5
    return {
        "alpha1": float(fitted[1]),
        "alpha1_err": float(errors[1]),
        "alpha2": float(fitted[2]),
        "alpha2_err": float(errors[2]),
        "f_break": fb,
        "f_break_err": float(fb * np.log(10.0) * errors[3]),
        "chi2": chi2,
        "dof": int(len(frequency) - 4),
        "interior": bool(interior and fitted[2] > fitted[1]),
    }


def analyze(burst, paths):
    detectors = [read_detector(path) for path in paths]
    summed = select_and_sum(detectors)
    start, stop = emission_window(summed)
    frequency, power, sum_var = periodogram(summed, start, stop)
    rebinned = log_rebin(frequency, power)
    net = rebinned["P"] - 2.0
    duration = float(summed["mid"][stop - 1] - summed["mid"][start])
    net_counts = float(np.sum(summed["net_rate"][start:stop]) * summed["dt"])
    return {
        **burst,
        "summed": summed,
        "window": (start, stop),
        "duration_obs": duration,
        "duration_rest": duration / (1.0 + burst["z"]),
        "net_counts": net_counts,
        "sum_var": sum_var,
        "f_obs": rebinned["f"],
        "f_lo": rebinned["f_lo"],
        "f_hi": rebinned["f_hi"],
        "P": rebinned["P"],
        "P_err": rebinned["P_err"],
        "P_net": net,
        "n_modes": rebinned["n"],
        "f_rest": rebinned["f"] * (1.0 + burst["z"]),
    }


def significant(result):
    keep = (result["P_net"] > result["P_err"]) & (result["n_modes"] >= 3)
    return keep


def common_rest_band(results):
    lows, highs = [], []
    for result in results:
        frequency = result["f_rest"][significant(result)]
        if len(frequency) < 4:
            continue
        lows.append(float(frequency[0]))
        highs.append(float(frequency[-1]))
    if not lows:
        raise RuntimeError("No burst has a significant power spectrum")
    low, high = max(lows), min(highs)
    if high <= low * 1.5:
        # Fall back to the band most of the bursts reach, rather than an empty overlap.
        low, high = float(np.median(lows)), float(np.median(highs))
    return low, high


def attach_fits(results, band):
    low, high = band
    for result in results:
        rest = result["f_rest"]
        own = significant(result)
        full = fit_powerlaw(rest[own], result["P_net"][own], result["P_err"][own])
        common = (rest >= low) & (rest <= high) & own
        shared = fit_powerlaw(rest[common], result["P_net"][common], result["P_err"][common])
        # Observer-frame index above the ~0.05 Hz break reported for the GBM average.
        observer = (result["f_obs"] >= 0.05) & (result["f_obs"] <= 2.0) & own
        obs_fit = fit_powerlaw(result["f_obs"][observer], result["P_net"][observer], result["P_err"][observer])
        broken = fit_broken(rest[own], result["P_net"][own], result["P_err"][own])
        prefer_break = False
        if broken and broken["interior"] and full is not None:
            # Two extra parameters. Require a clear improvement and a steeper high-frequency index.
            delta = full["chi2"] - broken["chi2"]
            relative = broken["f_break_err"] / broken["f_break"] if broken["f_break"] else np.inf
            # A break whose frequency is uncertain by a factor of two is not a measurement.
            prefer_break = (
                delta > 6.0
                and broken["alpha2"] > broken["alpha1"] + 0.2
                and np.isfinite(relative)
                and relative < 0.5
            )
        def inflate(fit):
            if fit and fit["dof"] > 0 and fit["chi2"] > fit["dof"]:
                fit["alpha_err"] *= float(np.sqrt(fit["chi2"] / fit["dof"]))
            return fit

        result["fit_full"] = inflate(full)
        result["fit_common"] = inflate(shared)
        result["fit_obs"] = inflate(obs_fit)
        result["fit_broken"] = broken if prefer_break else None
        result["fit_broken_raw"] = broken


def _format_alpha(fit):
    if fit is None or not np.isfinite(fit["alpha_err"]):
        return "—"
    return rf"{fit['alpha']:.2f} $\pm$ {fit['alpha_err']:.2f}"


def plot_spectra(results, band, path):
    """One rest-frame panel, sized for a single column (half the text width)."""
    low, high = band
    # 3.4 in matches the column figures in the LAT note, which are included at \columnwidth.
    style = {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 5.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "savefig.bbox": "tight",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.4, 3.15))
        xlim = (low / 1.4, high * 1.4)
        for result in results:
            frequency = result["f_rest"]
            fit = result["fit_common"]
            in_band = (frequency >= low) & (frequency <= high)
            keep = significant(result) & (frequency > 0)
            if fit is None:
                usable = keep & in_band
                scale = np.exp(np.mean(np.log(result["P_net"][usable]))) if np.any(usable) else 1.0
                label = result["name"]
            else:
                # Normalize by the power-law amplitude at the fit pivot so the slope is the shape.
                log_f = np.log10(frequency[keep])
                log_p = np.log10(result["P_net"][keep])
                sigma = result["P_err"][keep] / (result["P_net"][keep] * np.log(10))
                pivot = np.log10(fit["f0"])
                _slope, intercept = np.polyfit(log_f - pivot, log_p, 1, w=1.0 / sigma)
                scale = 10 ** intercept
                label = rf"GRB {result['name']}  $\alpha={fit['alpha']:.2f}\pm{fit['alpha_err']:.2f}$"
            shown = keep & (result["P_net"] > 0)
            emphasis = result["name"] == "260924A"
            ax.errorbar(
                frequency[shown], result["P_net"][shown] / scale,
                yerr=(result["P_err"][shown] / scale),
                fmt="o", ms=2.6 if emphasis else 2.1, lw=0.7 if emphasis else 0.5,
                color=COLORS[result["name"]], alpha=0.95,
                label=label, zorder=4 if emphasis else 3,
            )
            if fit is not None:
                grid = np.logspace(np.log10(fit["f_min"]), np.log10(fit["f_max"]), 40)
                model = (grid / fit["f0"]) ** (-fit["alpha"])
                ax.plot(grid, model, color=COLORS[result["name"]], lw=1.3 if emphasis else 0.9, zorder=4 if emphasis else 2)
        guide_f = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 20)
        pivot = np.sqrt(xlim[0] * xlim[1])
        ax.plot(guide_f, (guide_f / pivot) ** (-5.0 / 3.0), color="0.55", ls="--", lw=0.7, label=r"$\alpha=5/3$", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Rest-frame frequency (Hz)")
        ax.set_ylabel("Power, scaled to 1 at the pivot")
        ax.legend(loc="lower left", frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.7)
        fig.savefig(path, dpi=200)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def plot_lightcurves(results, path):
    fig, axes = plt.subplots(len(results), 1, figsize=(8.4, 11.0), sharex=False)
    for ax, result in zip(axes, results):
        summed = result["summed"]
        start, stop = result["window"]
        time = summed["mid"]
        ax.plot(time, summed["rate"], color=COLORS[result["name"]], lw=0.6, label="NaI sum")
        ax.plot(time, summed["background"], color="0.45", lw=1.0, label="Background fit")
        ax.axvspan(time[start], time[stop - 1], color=COLORS[result["name"]], alpha=0.08, lw=0)
        span = result["duration_obs"]
        pad = max(20.0, 0.25 * span)
        ax.set_xlim(time[start] - pad, time[stop - 1] + pad)
        ax.set_ylabel("counts s$^{-1}$")
        live = result["summed"]["live_fraction"]
        note = "deadtime" if result["summed"]["deadtime"] else f"live {live:.0%}"
        detectors = "+".join(result["summed"]["detectors"])
        ax.text(
            0.01, 0.86,
            f"GRB {result['name']}   z={result['z']:.3f}   {detectors}   {note}\n"
            f"window {result['duration_obs']:.0f} s observer, {result['duration_rest']:.0f} s rest",
            transform=ax.transAxes, fontsize=8, va="top",
        )
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    axes[-1].set_xlabel("Time since GBM trigger (s)")
    fig.suptitle("Light curves that enter the Fourier transform", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_tables(results, band):
    OUT.mkdir(parents=True, exist_ok=True)
    spectral_lines = ["burst,frame,f_Hz,f_lo,f_hi,P_leahy,P_err,P_net,n_modes"]
    for result in results:
        for frame, frequency in (("observer", result["f_obs"]), ("rest", result["f_rest"])):
            scale = (1.0 + result["z"]) if frame == "rest" else 1.0
            for f, lo, hi, p, err, net, modes in zip(
                frequency, result["f_lo"] * scale, result["f_hi"] * scale,
                result["P"], result["P_err"], result["P_net"], result["n_modes"],
            ):
                spectral_lines.append(
                    f"{result['name']},{frame},{f:.6e},{lo:.6e},{hi:.6e},{p:.6e},{err:.6e},{net:.6e},{int(modes)}"
                )
    (OUT / "pds_binned.csv").write_text("\n".join(spectral_lines) + "\n")

    header = (
        "burst,trigger,z,eiso_1e52,detectors,saturated_skipped,deadtime_limited,"
        "dt_s,emin_keV,emax_keV,t_start_s,t_stop_s,duration_obs_s,duration_rest_s,"
        "net_counts,min_livetime_fraction,background_order,"
        "alpha_common_rest,alpha_common_err,common_fmin_Hz,common_fmax_Hz,common_chi2,common_dof,"
        "alpha_full_rest,alpha_full_err,full_fmin_Hz,full_fmax_Hz,"
        "alpha_obs_0.05_2Hz,alpha_obs_err,"
        "break_detected,f_break_rest_Hz,f_break_err,alpha_low,alpha_high"
    )
    rows = [header]
    low, high = band
    for result in results:
        summed = result["summed"]
        start, stop = result["window"]
        common, full, obs = result["fit_common"], result["fit_full"], result["fit_obs"]
        broken = result["fit_broken"]

        def val(fit, key, fmt=".4f"):
            if fit is None or fit.get(key) is None or not np.isfinite(fit[key]):
                return ""
            return format(fit[key], fmt)

        rows.append(",".join([
            result["name"], result["trigger"], f"{result['z']:.4f}", f"{result['eiso']:.2f}",
            "+".join(summed["detectors"]),
            "+".join(summed["rejected_saturated"]),
            "yes" if summed["deadtime"] else "no",
            f"{summed['dt']:.3f}", f"{summed['energy'][0]:.1f}", f"{summed['energy'][1]:.1f}",
            f"{summed['mid'][start]:.3f}", f"{summed['mid'][stop - 1]:.3f}",
            f"{result['duration_obs']:.2f}", f"{result['duration_rest']:.2f}",
            f"{result['net_counts']:.1f}", f"{summed['live_fraction']:.3f}",
            str(summed["background_order"]),
            val(common, "alpha", ".3f"), val(common, "alpha_err", ".3f"),
            f"{low:.4f}", f"{high:.4f}",
            val(common, "chi2", ".2f"), val(common, "dof", ".0f"),
            val(full, "alpha", ".3f"), val(full, "alpha_err", ".3f"),
            val(full, "f_min", ".4f"), val(full, "f_max", ".4f"),
            val(obs, "alpha", ".3f"), val(obs, "alpha_err", ".3f"),
            "yes" if broken else "no",
            f"{broken['f_break']:.4f}" if broken else "",
            f"{broken['f_break_err']:.4f}" if broken else "",
            f"{broken['alpha1']:.3f}" if broken else "",
            f"{broken['alpha2']:.3f}" if broken else "",
        ]))
    (OUT / "pds_fits.csv").write_text("\n".join(rows) + "\n")


def self_test():
    """Poisson noise must sit at Leahy power 2, and a f^-1.7 series must fit near 1.7."""
    rng = np.random.default_rng(7)
    counts = rng.poisson(80.0, size=8192).astype(float)
    centered = counts - counts.mean()
    transform = np.fft.rfft(centered)
    power = 2.0 * np.abs(transform[1:-1]) ** 2 / counts.sum()
    noise = float(np.mean(power))
    if not 1.9 < noise < 2.1:
        raise RuntimeError(f"Leahy noise self-test gave {noise:.3f}, expected 2")

    n = 8192
    dt = 0.064
    frequency = np.fft.rfftfreq(n, dt)
    alpha = 1.70
    amplitude = np.zeros_like(frequency)
    amplitude[1:] = frequency[1:] ** (-alpha / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=len(frequency))
    phase[0] = 0.0
    series = np.fft.irfft(amplitude * np.exp(1j * phase), n=n)
    series = series - series.mean()
    transform = np.fft.rfft(series)
    # Any constant variance puts the same factor into every frequency, so the slope is unchanged.
    raw = np.abs(transform[1:-1]) ** 2
    freq = frequency[1:-1]
    rebinned = log_rebin(freq, raw)
    fitted = fit_powerlaw(rebinned["f"], rebinned["P"], rebinned["P_err"])
    if fitted is None or abs(fitted["alpha"] - alpha) > 0.12:
        got = None if fitted is None else fitted["alpha"]
        raise RuntimeError(f"Slope self-test recovered {got}, expected {alpha}")
    print(f"Self-test: Leahy noise {noise:.3f}, recovered slope {fitted['alpha']:.3f} (input {alpha})")


def main():
    self_test()
    DATA.mkdir(parents=True, exist_ok=True)
    print("Downloading NaI CTIME files")
    paths = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_burst, burst): burst["name"] for burst in BURSTS}
        for future in as_completed(futures):
            name = futures[future]
            paths[name] = future.result()
            print(f"  {name}: {len(paths[name])} detectors on disk")
    results = []
    for burst in BURSTS:
        print(f"Analyzing GRB {burst['name']}")
        result = analyze(burst, paths[burst["name"]])
        summed = result["summed"]
        print(
            f"  detectors {','.join(summed['detectors'])}  "
            f"skipped saturated {','.join(summed['rejected_saturated']) or '—'}  "
            f"live {summed['live_fraction']:.0%}  dt {summed['dt']} s  "
            f"window {result['duration_obs']:.1f} s  net counts {result['net_counts']:.0f}"
        )
        results.append(result)
    band = common_rest_band(results)
    print(f"Common rest-frame band: {band[0]:.3f}–{band[1]:.3f} Hz")
    attach_fits(results, band)
    for result in results:
        common = result["fit_common"]
        full = result["fit_full"]
        obs = result["fit_obs"]
        broken = result["fit_broken"]
        print(
            f"  GRB {result['name']}: "
            f"common α={None if common is None else round(common['alpha'], 2)} "
            f"± {None if common is None else round(common['alpha_err'], 2)}, "
            f"full-range α={None if full is None else round(full['alpha'], 2)}, "
            f"observer 0.05–2 Hz α={None if obs is None else round(obs['alpha'], 2)}, "
            f"break={'no' if broken is None else round(broken['f_break'], 3)}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    plot_spectra(results, band, OUT / "pds_comparison.png")
    plot_lightcurves(results, OUT / "lightcurves.png")
    write_tables(results, band)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
