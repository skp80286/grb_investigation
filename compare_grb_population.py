#!/usr/bin/env python3
"""Compare one GRB with the high-energy sample and the Fermi-GBM population.

Pass ``--grb 260924A``. The redshift is ``--redshift``, or the value in the
high-energy compilation when that burst is listed there. The GBM bcat is
``data/GRB<name>/gbm/glg_bcat_all_*.fit`` and the time-resolved Band table is
``results/GRB<name>/band_joint_gbm_bat_xrt.csv``. Override either with
``--bcat`` and ``--band-csv``. Figures go to ``results/GRB<name>/population``.
``--theta`` is the jet half-opening angle in degrees; without it the Frail
and Ghirlanda panels are skipped. With no arguments the script runs
GRB 260924A at z = 3.51 and theta = 1.7 deg.

The high-energy sample is the redshift-known compilation in
``high_energy_grb.csv`` (redshift, T90, isotropic energy in units of 10^52 erg,
and rest-frame peak energy). The general population is the HEASARC FERMIGBRST
catalog, the Fermi-GBM burst catalog of von Kienlin et al. (2020) with later
updates. The script downloads that catalog from the HEASARC TAP service and
caches it under ``data/catalogs/``.

Every pair of attributes is one scatter plot. Where the catalog separates
long and short bursts, only the long population is drawn and the power law
is fit to those bursts. The target burst is the black star, and the five other
bursts with the largest isotropic energies are marked in colour.

The same long-burst samples are also drawn as histograms, with this burst
marked on each axis. Pass ``--histograms-only`` to write those figures from
the cached catalogs and the parameter table without refitting.

The jet-angle panel places this burst on the Frail relation. The comparison
samples are Frail et al. (2001) Table 1 and Wang et al. (2018) Table 5,
read from the CSV files under ``data/catalogs/``. The Ghirlanda panel uses
that same Wang sample: collimation-corrected energy from Table 5, and
redshift and peak energy from Table 2.

T90, the 10-1000 keV fluence, and the 1024 ms peak photon flux come from the
GBM bcat file, which is the same pipeline that fills the catalog. T05 is
measured on the same 50-300 keV cumulative fluence as T90 and T50: it is the
time between the 47.5% and 52.5% points. Neither catalog tabulates T05, so it
is reported for this burst and is not a population axis. The isotropic energy
integrates the time-resolved joint GBM+BAT(+XRT) Band fits over the
observer-frame band that corresponds to 10 keV-10 MeV in the rest frame
(H0 = 67.7, Om = 0.3, flat). The peak energy is the nu F_nu peak of that
time-integrated model. Alpha and beta are the fluence-weighted means of the
same fits. Uncertainties on those derived quantities are the 16th and 84th
percentiles of a split-normal draw of the Band parameters.
"""

from __future__ import annotations

import argparse
import itertools
import urllib.parse
import urllib.request
import zlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io.votable import parse as parse_votable
import astropy.units as u

ROOT = Path(__file__).resolve().parent
# Running the script with no arguments repeats the GRB 260924A comparison.
DEFAULT_GRB = "260924A"
DEFAULT_REDSHIFT = 3.51
DEFAULT_THETA_DEG = 1.7
H0 = 67.7
OM0 = 0.3
HIGH_ENERGY_CSV = Path("/mnt/nas3_czti/grb_catalog_data/high_energy_grb.csv")
CACHE = ROOT / "data/catalogs/fermigbrst_population.csv"
PFLX_CACHE = ROOT / "data/catalogs/fermigbrst_pflx.csv"
FRAIL_TABLE = ROOT / "data/catalogs/frail2001_table1.csv"
WANG_TABLE = ROOT / "data/catalogs/wang2018_table5.csv"
WANG_COLOR = "#0f766e"
WANG_EDGE = "#115e59"
TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
N_TOP = 5

KEV_TO_ERG = 1.602176634e-9
PIVOT_KEV = 100.0
REST_EMIN_KEV = 10.0
REST_EMAX_KEV = 1.0e4

GBM_COLUMNS = (
    "name,trigger_name,t90,t90_error,fluence,fluence_error,"
    "flux_1024,flux_1024_error,"
    "flnc_band_epeak,flnc_band_epeak_pos_err,flnc_band_epeak_neg_err,"
    "flnc_band_alpha,flnc_band_alpha_pos_err,flnc_band_alpha_neg_err,"
    "flnc_band_beta,flnc_band_beta_pos_err,flnc_band_beta_neg_err,"
    "flnc_comp_epeak,flnc_comp_epeak_pos_err,flnc_comp_epeak_neg_err,"
    "flnc_comp_index,flnc_comp_index_pos_err,flnc_comp_index_neg_err"
)

# Earlier entries are placed on the x axis. The order follows the usual
# catalog figures: duration or redshift on x, and Eiso versus Ep last so the
# Amati panel has Eiso on x and Ep on y.
HIGH_ENERGY_ATTRS = ("z", "T90", "Eiso", "Ep")
GBM_ATTRS = ("T90", "fluence", "peak_flux", "Epeak", "alpha", "beta")

ATTR_LABEL = {
    "z": r"Redshift $z$",
    "T90": r"$T_{90}$ (s)",
    "Eiso": r"$E_{\mathrm{iso}}$ ($10^{52}$ erg)",
    "Ep": r"$E_{\mathrm{p,z}}$ (keV)",
    "fluence": r"Fluence, 10–1000 keV (erg cm$^{-2}$)",
    "peak_flux": r"Peak flux, 1024 ms, 10–1000 keV (ph cm$^{-2}$ s$^{-1}$)",
    "Epeak": r"$E_{\mathrm{peak}}$ (keV)",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "Lpeak": r"$L_{\mathrm{peak}}$ (erg s$^{-1}$)",
}
LOG_ATTRS = {"z", "T90", "Eiso", "Ep", "fluence", "peak_flux", "Epeak", "Lpeak"}
# A plotted point is kept when its catalog uncertainty is no larger than this.
# On a logarithmic axis the limit is a fraction of the measured value. On a
# linear axis (alpha, beta) it is an absolute limit. A missing uncertainty
# does not reject the point.
MAX_LOG_ERROR_FRACTION = 0.5
MAX_LINEAR_ERROR = 1.0
TOP_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e")
TOP_MARKERS = ("o", "s", "D", "^", "v")
POP_COLOR = "#c2410c"
POP_EDGE = "#7c2d12"
CLASS_STYLE = {
    "long": {"color": "#c2410c", "edge": "#7c2d12", "line": "#7c2d12", "band": "#c2410c", "label": "Long"},
    "short": {"color": "#0f766e", "edge": "#115e59", "line": "#115e59", "band": "#0f766e", "label": "Short"},
}


def _trapz(y, x):
    integrator = getattr(np, "trapezoid", None) or np.trapz
    return integrator(y, x)


def _norm_name(value):
    text = "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)
    text = text.upper().replace(" ", "")
    for prefix in ("GRB", "BN"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def _burst_rows(frame, grb_name):
    """Catalog rows for this burst, including the numeric GBM trigger on that date.

    Another burst on the same day, such as 260924B beside 260924A, is kept.
    """
    key = _norm_name(grb_name)
    norm = frame["norm"].astype(str)
    trigger = frame["trigger_norm"].astype(str)
    mask = (norm == key) | (trigger == key)
    if len(key) >= 6 and key[:6].isdigit():
        date = key[:6]
        letter = key[6:].upper()
        on_date = norm.str.startswith(date) | trigger.str.startswith(date)
        if letter:
            other = (
                norm.str.startswith(date)
                & ~norm.str.startswith(date + letter)
                & norm.str.slice(6, 7).str.fullmatch(r"[A-Z]", na=False)
            )
            mask = mask | (on_date & ~other)
        else:
            mask = mask | on_date
    return mask


def _draw_seed(grb_name):
    """Stable Monte Carlo seed. The first six digits are the GBM date."""
    key = _norm_name(grb_name)
    if len(key) >= 6 and key[:6].isdigit():
        return int(key[:6])
    return zlib.crc32(key.encode()) % 1_000_000


def resolve_target(args):
    """Paths and labels for the burst named by --grb."""
    name = _norm_name(args.grb)
    if not name:
        raise SystemExit("Pass a GRB name with --grb, for example 221009A.")
    bcat = args.bcat
    if bcat is None:
        found = sorted((ROOT / f"data/GRB{name}/gbm").glob("glg_bcat_all_*.fit"))
        bcat = found[-1] if found else None
    band = args.band_csv or (ROOT / f"results/GRB{name}/band_joint_gbm_bat_xrt.csv")
    theta = args.theta
    if theta is None and name == DEFAULT_GRB:
        theta = DEFAULT_THETA_DEG
    return SimpleNamespace(
        name=name,
        label=f"GRB {name}",
        bcat=None if bcat is None else Path(bcat),
        band=Path(band),
        output_dir=Path(args.output_dir) if args.output_dir else ROOT / f"results/GRB{name}/population",
        parameter_csv=f"grb{name.lower()}_parameters.csv",
        theta=theta,
        seed=_draw_seed(name),
    )


def resolve_redshift(redshift, name, sample):
    """Command-line redshift, else the high-energy compilation, else the built-in default."""
    if redshift is not None:
        return float(redshift)
    if sample is not None and "GRB" in sample.columns:
        hit = sample[sample["GRB"].map(_norm_name) == name]
        known = hit["z"][pd.to_numeric(hit["z"], errors="coerce") > 0]
        if len(known):
            return float(known.iloc[0])
    if name == DEFAULT_GRB:
        return DEFAULT_REDSHIFT
    raise SystemExit(
        f"GRB {name} has no redshift in the high-energy compilation. Pass --redshift."
    )


def download_gbm_catalog(cache_path, refresh=False):
    """Download the Fermi-GBM burst catalog and return it as a table."""
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path)
        print(f"Fermi-GBM catalog: {len(frame)} bursts from {cache_path}")
        return frame

    query = f"SELECT {GBM_COLUMNS} FROM fermigbrst"
    url = TAP + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query}
    )
    print("Downloading the Fermi-GBM burst catalog from HEASARC...")
    request = urllib.request.Request(url, headers={"User-Agent": "grb-investigation"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    table = parse_votable(BytesIO(payload)).get_first_table().to_table(use_names_over_ids=True).to_pandas()
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].map(lambda item: item.decode() if isinstance(item, bytes) else item)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    print(f"Fermi-GBM catalog: {len(table)} bursts, cached at {cache_path}")
    return table


def gbm_science_frame(catalog):
    """One row per burst, with the fluence-spectrum parameters used on the plots.

    The Band fit is kept when its peak energy, alpha, and beta are inside the
    range of converged catalog fits. Otherwise the Comptonized peak energy and
    index are used, and beta is left blank because that model has no beta.
    """
    frame = catalog.copy()
    for column in frame.columns:
        if column not in ("name", "trigger_name"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    band = frame["flnc_band_epeak"]
    band_ok = (
        band.between(1.0, 1.0e4)
        & frame["flnc_band_alpha"].between(-5.0, 5.0)
        & frame["flnc_band_beta"].between(-9.99, -0.05)
        & (frame["flnc_band_beta"] < frame["flnc_band_alpha"])
    )
    comp_ok = (
        frame["flnc_comp_epeak"].between(1.0, 1.0e4)
        & frame["flnc_comp_index"].between(-5.0, 5.0)
    )
    use_comp = ~band_ok & comp_ok

    def choose(band_col, comp_col):
        values = np.where(band_ok, frame[band_col], np.where(use_comp, frame[comp_col], np.nan))
        return values.astype(float)

    out = pd.DataFrame({
        "name": frame["name"].astype(str),
        "trigger_name": frame["trigger_name"].astype(str),
        "T90": frame["t90"],
        "T90_err": frame["t90_error"],
        "fluence": frame["fluence"],
        "fluence_err": frame["fluence_error"],
        "peak_flux": frame["flux_1024"],
        "peak_flux_err": frame["flux_1024_error"],
        "Epeak": choose("flnc_band_epeak", "flnc_comp_epeak"),
        "Epeak_lo": choose("flnc_band_epeak_neg_err", "flnc_comp_epeak_neg_err"),
        "Epeak_hi": choose("flnc_band_epeak_pos_err", "flnc_comp_epeak_pos_err"),
        "alpha": choose("flnc_band_alpha", "flnc_comp_index"),
        "alpha_lo": choose("flnc_band_alpha_neg_err", "flnc_comp_index_neg_err"),
        "alpha_hi": choose("flnc_band_alpha_pos_err", "flnc_comp_index_pos_err"),
        "beta": np.where(band_ok, frame["flnc_band_beta"], np.nan),
        "beta_lo": np.where(band_ok, frame["flnc_band_beta_neg_err"], np.nan),
        "beta_hi": np.where(band_ok, frame["flnc_band_beta_pos_err"], np.nan),
    })
    out["norm"] = out["name"].map(_norm_name)
    out["trigger_norm"] = out["trigger_name"].map(_norm_name)
    return out


def match_burst(frame, grb_name):
    """Match a GRB name such as 221009A or 090323 to a GBM catalog row."""
    key = _norm_name(grb_name)
    exact = frame[(frame["norm"] == key) | (frame["trigger_norm"] == key)]
    if len(exact):
        return exact.sort_values("fluence", ascending=False).iloc[0]
    if len(key) < 6 or not key[:6].isdigit():
        return None
    date = key[:6]
    letter = key[6:]
    mask = frame["norm"].str.startswith(date) | frame["trigger_norm"].str.startswith(date)
    candidates = frame[mask]
    if candidates.empty:
        return None
    if letter.isalpha():
        named = candidates[candidates["norm"].str.startswith(date + letter.upper())]
        if len(named):
            candidates = named
    return candidates.sort_values("fluence", ascending=False).iloc[0]


def read_bcat(path):
    header = fits.getheader(path, 0)
    return {
        "T90": float(header["T90"]),
        "T90_err": float(header["T90_ERR"]),
        "T50": float(header["T50"]),
        "T50_err": float(header["T50_ERR"]),
        "fluence": float(header["FLU"]),
        "fluence_err": float(header["FLU_ERR"]),
        "peak_flux": float(header["PFLX"]),
        "peak_flux_err": float(header["PFLX_ERR"]),
    }


def _header_interval(value):
    text = str(value).strip().strip("()")
    left, right = (float(part) for part in text.split(","))
    return left, right


def _fluence_crossing(time, cumulative, level):
    """First time the cumulative fluence reaches level, linearly interpolated."""
    above = np.where(cumulative >= level)[0]
    if len(above) == 0:
        return np.nan
    index = int(above[0])
    if index == 0:
        return float(time[0])
    y0, y1 = float(cumulative[index - 1]), float(cumulative[index])
    if y1 == y0:
        return float(time[index])
    fraction = (level - y0) / (y1 - y0)
    return float(time[index - 1] + fraction * (time[index] - time[index - 1]))


def cumulative_durations(path, n_draws=800, seed=1):
    """T90, T50, and T05 from the bcat 50-300 keV cumulative photon fluence.

    The curve is the background-subtracted fluence stored for the duration
    calculation, accumulated from the end of the pre-burst background window
    to the start of the post-burst window. T90 runs from 5% to 95% of that
    fluence, T50 from 25% to 75%, and T05 from 47.5% to 52.5%. The uncertainty
    is the 16th-84th percentile span after perturbing each time bin by its
    fluence error.
    """
    header = fits.getheader(path, 0)
    table = fits.getdata(path, "FIT PARAMS")
    time_bin = np.array(table["TIMEBIN"].tolist(), dtype=float)
    duration_fluence = np.array(table["DURFLNC"].tolist(), dtype=float)
    pre_end = _header_interval(header["LOBCKINT"])[1]
    post_start = _header_interval(header["HIBCKINT"])[0]
    inside = (time_bin[:, 1] > pre_end) & (time_bin[:, 0] < post_start)
    time = time_bin[inside, 1]
    cumulative = duration_fluence[inside, 0]
    increment_error = duration_fluence[inside, 1]
    signal = cumulative - cumulative[0]
    increment = np.diff(signal, prepend=0.0)

    def measure(curve):
        total = float(curve[-1])
        if not np.isfinite(total) or total <= 0:
            return None
        measured = {}
        for name, low, high in (("T90", 0.05, 0.95), ("T50", 0.25, 0.75), ("T05", 0.475, 0.525)):
            start = _fluence_crossing(time, curve, low * total)
            stop = _fluence_crossing(time, curve, high * total)
            measured[name] = (stop - start, start, stop)
        return measured

    best = measure(signal)
    if best is None:
        raise RuntimeError("The bcat duration fluence does not rise between the background windows.")

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_draws) for name in ("T90", "T50", "T05")}
    for i in range(n_draws):
        noisy = np.cumsum(increment + rng.normal(0.0, increment_error))
        sample = measure(noisy - noisy[0])
        for name in draws:
            draws[name][i] = np.nan if sample is None else sample[name][0]
    errors = {}
    for name, (duration, start, stop) in best.items():
        sample = draws[name][np.isfinite(draws[name])]
        if len(sample) < 100:
            raise RuntimeError(f"Only {len(sample)} valid draws for {name}.")
        low, high = np.percentile(sample, [16, 84])
        errors[name] = (float(duration), max(float(duration - low), 0.0), max(float(high - duration), 0.0), float(start), float(stop))
    return errors


def _split_draw(rng, center, sig_lo, sig_hi, n):
    draw = rng.standard_normal(n)
    sigma = np.where(draw < 0.0, np.abs(sig_lo), np.abs(sig_hi))
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return np.asarray(center, float) + draw * sigma


def _band_photons(K, alpha, xp, beta, energy):
    K = np.asarray(K, float)
    alpha = np.asarray(alpha, float)
    xp = np.asarray(xp, float)
    beta = np.asarray(beta, float)
    E0 = xp / (2.0 + alpha)
    break_energy = (alpha - beta) * E0
    valid = (
        np.isfinite(K) & (K > 0)
        & np.isfinite(alpha) & (alpha > -1.999)
        & np.isfinite(xp) & (xp > 1.0)
        & np.isfinite(beta) & (beta < alpha)
        & np.isfinite(E0) & (E0 > 0)
        & np.isfinite(break_energy) & (break_energy > 0)
    )
    photons = np.zeros((len(K), len(energy)))
    if not np.any(valid):
        return photons, valid
    ee = energy[None, :]
    br = break_energy[valid, None]
    low = ee < br
    Kv = K[valid, None]
    av = alpha[valid, None]
    bv = beta[valid, None]
    E0v = E0[valid, None]
    block = np.empty((int(valid.sum()), len(energy)))
    block[low] = (Kv * (ee / PIVOT_KEV) ** av * np.exp(-ee / E0v))[low]
    if np.any(~low):
        factor = np.exp(bv - av) * (br / PIVOT_KEV) ** (av - bv)
        block[~low] = (Kv * factor * (ee / PIVOT_KEV) ** bv)[~low]
    photons[valid] = block
    return photons, valid


def this_grb_parameters(bcat_path, band_path, redshift, n_draws=800, seed=1):
    """T90, fluence, and flux from the bcat; Eiso and spectral shape from the Band fits."""
    measured = read_bcat(bcat_path)
    durations = cumulative_durations(bcat_path, n_draws=n_draws, seed=seed)
    bands = pd.read_csv(band_path)
    energy = np.logspace(np.log10(1.0), np.log10(2.0e4), 4000)
    in_gbm = (energy >= 10.0) & (energy <= 1000.0)
    rest_emin = REST_EMIN_KEV / (1.0 + redshift)
    rest_emax = REST_EMAX_KEV / (1.0 + redshift)
    in_bol = (energy >= rest_emin) & (energy <= rest_emax)

    def spectrum(K, alpha, xp, beta):
        photons, valid = _band_photons(K, alpha, xp, beta, energy)
        fluence_density = photons * energy * KEV_TO_ERG
        return fluence_density, valid

    K = bands["K"].to_numpy(float)
    alpha = bands["alpha"].to_numpy(float)
    xp = bands["xp"].to_numpy(float)
    beta = bands["beta"].to_numpy(float)
    dt = (bands["t_end"] - bands["t_start"]).to_numpy(float)

    best_density, best_valid = spectrum(K, alpha, xp, beta)
    if not np.all(best_valid):
        raise RuntimeError("A best-fit Band spectrum is outside the range where Ep is defined.")
    best_fluence = (best_density * dt[:, None]).sum(axis=0)
    # best_fluence is N(E) * E, so one more factor of E gives nu F_nu.
    ep_obs = float(energy[np.argmax(energy * best_fluence)])
    s_bol = float(_trapz(best_fluence[in_bol], energy[in_bol]))
    s10_bins = np.array([
        float(_trapz(best_density[i, in_gbm], energy[in_gbm])) * dt[i] for i in range(len(dt))
    ])
    alpha_w = float(np.sum(s10_bins * alpha) / np.sum(s10_bins))
    beta_w = float(np.sum(s10_bins * beta) / np.sum(s10_bins))

    cosmo = FlatLambdaCDM(H0=H0, Om0=OM0)
    d_l = cosmo.luminosity_distance(redshift).to(u.cm).value
    eiso_erg = 4.0 * np.pi * d_l ** 2 * s_bol / (1.0 + redshift)

    rng = np.random.default_rng(seed)
    n_bins = len(bands)
    drawn = {
        name: np.vstack([
            _split_draw(rng, bands[name].iloc[i], bands[f"{name}_err_lo"].iloc[i], bands[f"{name}_err_hi"].iloc[i], n_draws)
            for i in range(n_bins)
        ])
        for name in ("K", "alpha", "xp", "beta")
    }
    eiso_draws = np.empty(n_draws)
    ep_draws = np.empty(n_draws)
    alpha_draws = np.empty(n_draws)
    beta_draws = np.empty(n_draws)
    for draw in range(n_draws):
        density, valid = spectrum(drawn["K"][:, draw], drawn["alpha"][:, draw], drawn["xp"][:, draw], drawn["beta"][:, draw])
        if not np.all(valid):
            eiso_draws[draw] = ep_draws[draw] = alpha_draws[draw] = beta_draws[draw] = np.nan
            continue
        fluence = (density * dt[:, None]).sum(axis=0)
        ep_draws[draw] = energy[np.argmax(energy * fluence)]
        s_bol_draw = float(_trapz(fluence[in_bol], energy[in_bol]))
        eiso_draws[draw] = 4.0 * np.pi * d_l ** 2 * s_bol_draw / (1.0 + redshift)
        s10 = np.array([float(_trapz(density[i, in_gbm], energy[in_gbm])) * dt[i] for i in range(n_bins)])
        weight = np.clip(s10, 0, None)
        if weight.sum() <= 0:
            alpha_draws[draw] = beta_draws[draw] = np.nan
            continue
        alpha_draws[draw] = np.sum(weight * drawn["alpha"][:, draw]) / weight.sum()
        beta_draws[draw] = np.sum(weight * drawn["beta"][:, draw]) / weight.sum()

    def percentile_errors(center, sample):
        sample = sample[np.isfinite(sample)]
        if len(sample) < 100:
            raise RuntimeError(f"Only {len(sample)} valid Monte Carlo draws; the Band error propagation failed.")
        low, high = np.percentile(sample, [16, 84])
        return max(float(center - low), 0.0), max(float(high - center), 0.0)

    eiso_lo, eiso_hi = percentile_errors(eiso_erg, eiso_draws)
    ep_lo, ep_hi = percentile_errors(ep_obs, ep_draws)
    alpha_lo, alpha_hi = percentile_errors(alpha_w, alpha_draws)
    beta_lo, beta_hi = percentile_errors(beta_w, beta_draws)
    n_valid = int(np.isfinite(eiso_draws).sum())

    values = {
        "z": (redshift, 0.0, 0.0),
        "T90": (measured["T90"], measured["T90_err"], measured["T90_err"]),
        "T05": durations["T05"][:3],
        "Eiso": (eiso_erg / 1.0e52, eiso_lo / 1.0e52, eiso_hi / 1.0e52),
        "Ep": (ep_obs * (1.0 + redshift), ep_lo * (1.0 + redshift), ep_hi * (1.0 + redshift)),
        "fluence": (measured["fluence"], measured["fluence_err"], measured["fluence_err"]),
        "peak_flux": (measured["peak_flux"], measured["peak_flux_err"], measured["peak_flux_err"]),
        "Epeak": (ep_obs, ep_lo, ep_hi),
        "alpha": (alpha_w, alpha_lo, alpha_hi),
        "beta": (beta_w, beta_lo, beta_hi),
    }
    info = {
        "eiso_erg": eiso_erg,
        "s_bol": s_bol,
        "s10_model": float(s10_bins.sum()),
        "s10_bcat": measured["fluence"],
        "n_valid": n_valid,
        "n_draws": n_draws,
        "T05_start": durations["T05"][3],
        "T05_stop": durations["T05"][4],
        "T90_from_curve": durations["T90"][0],
        "T50_from_curve": durations["T50"][0],
        "T50_catalog": measured["T50"],
    }
    return values, info


def top_energy_bursts(sample, n_top):
    ranked = sample.sort_values("Eiso", ascending=False)
    # This burst is not in the compilation. "Other" is every burst in the file.
    return ranked.head(n_top).copy()


def _positive_error(value, err_lo, err_hi, log_axis):
    err_lo = 0.0 if not np.isfinite(err_lo) else max(float(err_lo), 0.0)
    err_hi = 0.0 if not np.isfinite(err_hi) else max(float(err_hi), 0.0)
    if log_axis and value > 0:
        err_lo = min(err_lo, value * 0.999)
    return err_lo, err_hi


def _population_errors(frame, attr):
    """Return (value, err_lo, err_hi) arrays. Missing errors become zero."""
    if attr == "z":
        value = frame["z"].to_numpy(float)
        return value, np.zeros_like(value), np.zeros_like(value)
    if attr == "T90":
        value = frame["T90"].to_numpy(float)
        err = frame["T90_err"].to_numpy(float) if "T90_err" in frame else np.zeros(len(frame))
        return value, err, err
    if attr == "Eiso":
        value = frame["Eiso"].to_numpy(float)
        err = frame["Eiso_err"].to_numpy(float)
        return value, err, err
    if attr == "Ep":
        value = frame["Ep"].to_numpy(float)
        err = frame["Ep_err"].to_numpy(float)
        return value, err, err
    value = frame[attr].to_numpy(float)
    lo = frame[f"{attr}_lo"].to_numpy(float) if f"{attr}_lo" in frame else frame.get(f"{attr}_err", pd.Series(0, index=frame.index)).to_numpy(float)
    hi = frame[f"{attr}_hi"].to_numpy(float) if f"{attr}_hi" in frame else lo
    if f"{attr}_err" in frame and f"{attr}_lo" not in frame:
        err = frame[f"{attr}_err"].to_numpy(float)
        lo = hi = err
    return value, np.asarray(lo, float), np.asarray(hi, float)


def _precise_enough(value, err_lo, err_hi, log_axis):
    """True where the catalog uncertainty is within the plotting limit."""
    lo = np.asarray(err_lo, float)
    hi = np.asarray(err_hi, float)
    lo = np.where(np.isfinite(lo) & (lo > 0), lo, 0.0)
    hi = np.where(np.isfinite(hi) & (hi > 0), hi, 0.0)
    worst = np.maximum(lo, hi)
    if log_axis:
        return worst <= MAX_LOG_ERROR_FRACTION * np.abs(np.asarray(value, float))
    return worst <= MAX_LINEAR_ERROR


def _error_pair(value, err_lo, err_hi, log_axis):
    """Asymmetric errors for errorbar. None when the catalog has no uncertainties."""
    err_lo = np.asarray(err_lo, float).copy()
    err_hi = np.asarray(err_hi, float).copy()
    err_lo[~np.isfinite(err_lo) | (err_lo < 0)] = 0.0
    err_hi[~np.isfinite(err_hi) | (err_hi < 0)] = 0.0
    if log_axis:
        positive = np.asarray(value, float) > 0
        err_lo[positive] = np.minimum(err_lo[positive], np.asarray(value, float)[positive] * 0.999)
    if not np.any(err_lo > 0) and not np.any(err_hi > 0):
        return None
    return [err_lo, err_hi]


def _lock_view(ax):
    """Keep the axes on the points already drawn, so later artists stay inside the frame."""
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_autoscalex_on(False)
    ax.set_autoscaley_on(False)
    return (xmin, xmax), (ymin, ymax)


def burst_class(value):
    """Map a catalog type to long or short. Short-with-extended-emission counts as short."""
    text = "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value).strip().upper()
    if text in {"L", "LONG"}:
        return "long"
    if text in {"S", "SHORT", "S+EE"}:
        return "short"
    return ""


def duration_class(t90):
    """GBM class from the catalog T90. The standard divider is 2 s."""
    if not np.isfinite(t90) or t90 <= 0:
        return ""
    return "short" if float(t90) < 2.0 else "long"


def _style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.grid(False, which="minor")
    ax.grid(True, which="major", color="0.9", lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def _draw_relation(ax, grid, x_line, intercept, slope, sigma, ylog, band_color, line_color, line_label, label_bands):
    fitted = intercept + slope * grid
    for nsig, alpha in ((3, 0.10), (2, 0.14), (1, 0.22)):
        low = fitted - nsig * sigma
        high = fitted + nsig * sigma
        y_low = 10 ** low if ylog else low
        y_high = 10 ** high if ylog else high
        band_label = rf"${nsig}\sigma$" if label_bands else None
        ax.fill_between(x_line, y_low, y_high, color=band_color, alpha=alpha, lw=0, zorder=0, label=band_label)
    y_line = 10 ** fitted if ylog else fitted
    ax.plot(x_line, y_line, color=line_color, lw=1.4, zorder=2, label=line_label)


def _fit_slope(design, ordinate):
    """Ordinary least-squares slope and its standard error.

    The last column of design is the term whose coefficient is the slope.
    The uncertainty is the residual-scaled standard error of that coefficient.
    """
    beta, sse, rank, _ = np.linalg.lstsq(design, ordinate, rcond=None)
    n_points, n_parameters = design.shape
    variance = np.nan
    if sse.size and rank == n_parameters and n_points > n_parameters:
        sigma2 = float(sse[0]) / (n_points - n_parameters)
        try:
            covariance = sigma2 * np.linalg.inv(design.T @ design)
            variance = float(covariance[-1, -1])
        except np.linalg.LinAlgError:
            variance = np.nan
    error = float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else np.nan
    return beta, error


def _fit_label(prefix, slope, slope_err, power):
    """Legend entry giving the fitted index and its uncertainty.

    The index is the slope in the coordinates of the fit: log10 of an axis
    when that axis is logarithmic, and the plotted value when it is linear.
    """
    if np.isfinite(slope_err) and slope_err > 0:
        places = int(np.clip(np.ceil(-np.log10(abs(slope_err))) + 1, 0, 4))
        index = rf"${slope:.{places}f} \pm {slope_err:.{places}f}$"
    else:
        index = rf"${slope:.2f}$"
    kind = "power law" if power else "fit"
    return f"{prefix}{kind}, index {index}"


def add_relation_fit(ax, x, y, xlog, ylog, classes=None):
    """Power laws across the full axes, with 1, 2 and 3 sigma bands.

    When both a long and a short class are present, the two lines share one
    index and have separate normalizations. Each band is that class's own
    residual scatter.     Otherwise one line is fit to every point. The legend gives the fitted
    index and its standard error for every line, including fits that are
    not a power law because one axis is linear.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if classes is None:
        classes = np.array([""] * len(x))
    classes = np.asarray(classes)
    ok = np.isfinite(x) & np.isfinite(y)
    if xlog:
        ok &= x > 0
    if ylog:
        ok &= y > 0
    x, y, classes = x[ok], y[ok], classes[ok]
    if len(x) < 5:
        return
    abscissa = np.log10(x) if xlog else x
    ordinate = np.log10(y) if ylog else y
    if np.std(abscissa) == 0:
        return
    long = classes == "long"
    short = classes == "short"
    shared = int(long.sum()) >= 8 and int(short.sum()) >= 8
    if shared:
        design = np.column_stack([long.astype(float), short.astype(float), abscissa])
        use = long | short
        beta, slope_err = _fit_slope(design[use], ordinate[use])
        intercept_long, intercept_short, slope = (float(value) for value in beta)
        groups = (
            ("long", intercept_long, long),
            ("short", intercept_short, short),
        )
    else:
        design = np.column_stack([np.ones(len(abscissa)), abscissa])
        beta, slope_err = _fit_slope(design, ordinate)
        intercept, slope = (float(value) for value in beta)
        groups = (("all", intercept, np.ones(len(abscissa), dtype=bool)),)
    (xmin, xmax), (ymin, ymax) = _lock_view(ax)
    if xlog and (xmin <= 0 or xmax <= 0):
        return
    grid = np.linspace(np.log10(xmin), np.log10(xmax), 400) if xlog else np.linspace(xmin, xmax, 400)
    x_line = 10 ** grid if xlog else grid
    power = xlog and ylog
    for name, intercept, mask in groups:
        residual = ordinate[mask] - (intercept + slope * abscissa[mask])
        sigma = float(np.std(residual, ddof=2))
        if not np.isfinite(sigma) or sigma <= 0 or int(mask.sum()) < 5:
            continue
        if name == "all":
            band_color = line_color = "#1f4e79"
            line_label = _fit_label("", slope, slope_err, power)
            label_bands = True
        else:
            band_color = CLASS_STYLE[name]["band"]
            line_color = CLASS_STYLE[name]["line"]
            line_label = _fit_label(f"{CLASS_STYLE[name]['label']} ", slope, slope_err, power)
            label_bands = False
        _draw_relation(ax, grid, x_line, intercept, slope, sigma, ylog, band_color, line_color, line_label, label_bands)
    if shared:
        print(f"  shared index {slope:.3f} ± {slope_err:.3f}: long {int(long.sum())}, short {int(short.sum())}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_autoscalex_on(False)
    ax.set_autoscaley_on(False)


def plot_pair(path, xlabel_key, ylabel_key, population, highlights, this_grb, population_label, star_label):
    x, xlo, xhi = _population_errors(population, xlabel_key)
    y, ylo, yhi = _population_errors(population, ylabel_key)
    keep = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    if xlabel_key in LOG_ATTRS:
        keep &= x > 0
    if ylabel_key in LOG_ATTRS:
        keep &= y > 0
    long_only = False
    if "class" in population.columns:
        class_all = population["class"].astype(str).to_numpy()
        if np.isin(class_all, ["long", "short"]).any():
            keep &= class_all == "long"
            long_only = True
    before_error_cut = int(keep.sum())
    keep &= _precise_enough(x, xlo, xhi, xlabel_key in LOG_ATTRS)
    keep &= _precise_enough(y, ylo, yhi, ylabel_key in LOG_ATTRS)
    if int(keep.sum()) < before_error_cut:
        print(
            f"  error cut {xlabel_key} vs {ylabel_key}: "
            f"{int(keep.sum())} of {before_error_cut} kept"
        )
    x, y = x[keep], y[keep]
    xlo, xhi = xlo[keep], xhi[keep]
    ylo, yhi = ylo[keep], yhi[keep]
    if "class" in population.columns:
        classes = population["class"].astype(str).to_numpy()[keep]
    else:
        classes = np.array([""] * len(x))

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    known = np.isin(classes, list(CLASS_STYLE))
    if known.any():
        for name, style in CLASS_STYLE.items():
            mask = classes == name
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask], y[mask], s=22, c=style["color"], alpha=0.9, linewidths=0.35,
                edgecolors=style["edge"], rasterized=True, zorder=2 if name == "long" else 3,
                label=f"{style['label']} ({int(mask.sum())})",
            )
        if np.any(~known):
            ax.scatter(
                x[~known], y[~known], s=22, c="#9aa0a6", alpha=0.8, linewidths=0,
                rasterized=True, zorder=2, label=f"Unclassified ({int((~known).sum())})",
            )
    else:
        ax.scatter(
            x, y, s=22, c=POP_COLOR, alpha=0.9, linewidths=0.35, edgecolors=POP_EDGE,
            rasterized=True, zorder=2, label=f"{population_label} ({len(x)})",
        )

    def draw_point(record, marker, color, size, label, zorder):
        xv, xelo, xehi = record[xlabel_key]
        yv, yelo, yehi = record[ylabel_key]
        if not (np.isfinite(xv) and np.isfinite(yv)):
            return False
        if xlabel_key in LOG_ATTRS and xv <= 0:
            return False
        if ylabel_key in LOG_ATTRS and yv <= 0:
            return False
        xelo, xehi = _positive_error(xv, xelo, xehi, xlabel_key in LOG_ATTRS)
        yelo, yehi = _positive_error(yv, yelo, yehi, ylabel_key in LOG_ATTRS)
        ax.errorbar(
            [xv], [yv], xerr=[[xelo], [xehi]], yerr=[[yelo], [yehi]],
            fmt="none", ecolor=color, elinewidth=0.8, capsize=2, zorder=zorder,
        )
        ax.scatter(
            [xv], [yv], s=size, marker=marker, c=color, edgecolors="black",
            linewidths=0.45, zorder=zorder + 1, label=label,
        )
        return True

    for index, item in enumerate(highlights):
        if long_only and item.get("class") != "long":
            continue
        draw_point(
            item["values"], TOP_MARKERS[index], TOP_COLORS[index], 58,
            item["label"], 4,
        )
    draw_point(this_grb, "*", "black", 220, star_label, 6)

    if xlabel_key in LOG_ATTRS:
        ax.set_xscale("log")
    if ylabel_key in LOG_ATTRS:
        ax.set_yscale("log")
    if xlabel_key == "T90":
        ax.axvline(2.0, color="0.45", lw=0.7, ls="--", zorder=0)
        ax.plot([], [], color="0.45", lw=0.7, ls="--", label=r"$T_{90}=2$ s")
    if ylabel_key == "T90":
        ax.axhline(2.0, color="0.45", lw=0.7, ls="--", zorder=0)
        ax.plot([], [], color="0.45", lw=0.7, ls="--", label=r"$T_{90}=2$ s")

    _lock_view(ax)
    xerr = _error_pair(x, xlo, xhi, xlabel_key in LOG_ATTRS)
    yerr = _error_pair(y, ylo, yhi, ylabel_key in LOG_ATTRS)
    if xerr is not None or yerr is not None:
        if known.any():
            groups = [(classes == name, CLASS_STYLE[name]["color"]) for name in CLASS_STYLE]
            groups.append((~known, "#9aa0a6"))
        else:
            groups = [(np.ones(len(x), dtype=bool), POP_COLOR)]
        for mask, color in groups:
            if not np.any(mask):
                continue
            ax.errorbar(
                x[mask], y[mask],
                xerr=None if xerr is None else [xerr[0][mask], xerr[1][mask]],
                yerr=None if yerr is None else [yerr[0][mask], yerr[1][mask]],
                fmt="none", ecolor=color, elinewidth=0.45, capsize=0, alpha=0.4,
                zorder=1, rasterized=True,
            )

    fit_x, fit_y, fit_class = list(x), list(y), list(classes)
    for item in highlights:
        if long_only and item.get("class") != "long":
            continue
        record = item["values"]
        if xlabel_key not in record or ylabel_key not in record:
            continue
        fit_x.append(record[xlabel_key][0])
        fit_y.append(record[ylabel_key][0])
        fit_class.append(item.get("class", ""))
    add_relation_fit(ax, fit_x, fit_y, xlabel_key in LOG_ATTRS, ylabel_key in LOG_ATTRS, fit_class)

    ax.set_xlabel(ATTR_LABEL[xlabel_key])
    ax.set_ylabel(ATTR_LABEL[ylabel_key])
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.94, edgecolor="none", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def high_energy_record(row):
    return {
        "z": (float(row["z"]), 0.0, 0.0),
        "T90": (float(row["T90"]), np.nan, np.nan),
        "Eiso": (float(row["Eiso"]), float(row["Eiso_err"]), float(row["Eiso_err"])),
        "Ep": (float(row["Ep"]), float(row["Ep_err"]), float(row["Ep_err"])),
    }


def gbm_record(row):
    def triple(name, err_name=None):
        if err_name is None:
            return (float(row[name]), float(row[f"{name}_lo"]), float(row[f"{name}_hi"]))
        err = float(row[err_name])
        return (float(row[name]), err, err)

    return {
        "T90": triple("T90", "T90_err"),
        "fluence": triple("fluence", "fluence_err"),
        "peak_flux": triple("peak_flux", "peak_flux_err"),
        "Epeak": triple("Epeak"),
        "alpha": triple("alpha"),
        "beta": triple("beta"),
    }


PFLX_COLUMNS = (
    "name,trigger_name,pflx_band_epeak,pflx_band_alpha,pflx_band_beta,pflx_band_ergflux,"
    "pflx_comp_epeak,pflx_comp_index,pflx_comp_ergflux"
)


def download_pflx_catalog(cache_path, refresh=False):
    """Peak-flux spectra from FERMIGBRST, used for the Yonetoku luminosity."""
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path)
        print(f"GBM peak-flux spectra: {len(frame)} bursts from {cache_path}")
        return frame
    query = f"SELECT {PFLX_COLUMNS} FROM fermigbrst"
    url = TAP + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query}
    )
    print("Downloading Fermi-GBM peak-flux spectra from HEASARC...")
    request = urllib.request.Request(url, headers={"User-Agent": "grb-investigation"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    table = parse_votable(BytesIO(payload)).get_first_table().to_table(use_names_over_ids=True).to_pandas()
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].map(lambda item: item.decode() if isinstance(item, bytes) else item)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    print(f"GBM peak-flux spectra: {len(table)} bursts, cached at {cache_path}")
    return table


def _bolometric_ratio(alpha, epeak, beta, redshift, energy):
    """Energy-flux in the rest-frame 10 keV-10 MeV band, divided by the 10-1000 keV flux."""
    photons, valid = _band_photons([1.0], [alpha], [epeak], [beta], energy)
    if not valid[0]:
        return np.nan
    spectrum = photons[0] * energy
    observed = (energy >= 10.0) & (energy <= 1000.0)
    rest = (energy >= REST_EMIN_KEV / (1.0 + redshift)) & (energy <= REST_EMAX_KEV / (1.0 + redshift))
    denominator = _trapz(spectrum[observed], energy[observed])
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(_trapz(spectrum[rest], energy[rest]) / denominator)


def _peak_shape(row):
    """Band peak-flux spectrum, or the Comptonized spectrum when Band did not converge."""
    epeak = row["pflx_band_epeak"]
    alpha = row["pflx_band_alpha"]
    beta = row["pflx_band_beta"]
    flux = row["pflx_band_ergflux"]
    band_ok = (
        np.isfinite(epeak) and 1.0 < epeak < 1.0e4
        and np.isfinite(alpha) and -5.0 < alpha < 5.0
        and np.isfinite(beta) and -9.99 < beta < -0.05 and beta < alpha
        and np.isfinite(flux) and flux > 0
    )
    if band_ok:
        return float(alpha), float(epeak), float(beta), float(flux)
    epeak = row["pflx_comp_epeak"]
    alpha = row["pflx_comp_index"]
    flux = row["pflx_comp_ergflux"]
    comp_ok = (
        np.isfinite(epeak) and 1.0 < epeak < 1.0e4
        and np.isfinite(alpha) and -1.999 < alpha < 5.0
        and np.isfinite(flux) and flux > 0
    )
    if comp_ok:
        return float(alpha), float(epeak), -10.0, float(flux)
    return None


def luminosity_from_peak_flux(alpha, epeak, beta, flux_10_1000, redshift, distance_cm, energy):
    ratio = _bolometric_ratio(alpha, epeak, beta, redshift, energy)
    if not np.isfinite(ratio):
        return np.nan
    return 4.0 * np.pi * distance_cm ** 2 * flux_10_1000 * ratio


def this_burst_peak_luminosity(bcat_path, band_path, redshift, n_draws=800, seed=1):
    """1.024 s peak luminosity from the bcat photon flux and the joint Band fit.

    The GBM catalog has no peak-flux spectrum for this burst. The 1024 ms peak
    photon flux is in the bcat header, and it falls in one time-resolved joint
    Band interval. That interval's shape converts the photon flux to the
    rest-frame 10 keV-10 MeV energy flux.
    """
    header = fits.getheader(bcat_path, 0)
    photon_flux = float(header["PFLX"])
    photon_flux_err = float(header["PFLX_ERR"])
    peak_time = float(header["PFLX_BEG"]) + 0.5 * float(header["PFLX_INT"])
    bands = pd.read_csv(band_path)
    row = bands[(bands["t_start"] <= peak_time) & (bands["t_end"] >= peak_time)].iloc[0]
    energy = np.logspace(np.log10(0.2), np.log10(2.0e4), 4000)
    photons, valid = _band_photons([row["K"]], [row["alpha"]], [row["xp"]], [row["beta"]], energy)
    if not valid[0]:
        raise RuntimeError("The Band interval containing the peak flux has no peak energy.")
    observed = (energy >= 10.0) & (energy <= 1000.0)
    model_photon_flux = float(_trapz(photons[0, observed], energy[observed]))
    scale = photon_flux / model_photon_flux
    distance = FlatLambdaCDM(H0=H0, Om0=OM0).luminosity_distance(redshift).to(u.cm).value
    rest = (energy >= REST_EMIN_KEV / (1.0 + redshift)) & (energy <= REST_EMAX_KEV / (1.0 + redshift))
    energy_flux = float(_trapz(photons[0, rest] * energy[rest], energy[rest]) * KEV_TO_ERG) * scale
    luminosity = 4.0 * np.pi * distance ** 2 * energy_flux

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        alpha = _split_draw(rng, row["alpha"], row["alpha_err_lo"], row["alpha_err_hi"], 1)[0]
        xp = _split_draw(rng, row["xp"], row["xp_err_lo"], row["xp_err_hi"], 1)[0]
        beta = _split_draw(rng, row["beta"], row["beta_err_lo"], row["beta_err_hi"], 1)[0]
        flux = photon_flux + rng.normal(0.0, photon_flux_err)
        if flux <= 0 or xp <= 1 or alpha <= -1.999 or beta >= alpha:
            draws[i] = np.nan
            continue
        trial, ok = _band_photons([1.0], [alpha], [xp], [beta], energy)
        if not ok[0]:
            draws[i] = np.nan
            continue
        photon_den = float(_trapz(trial[0, observed], energy[observed]))
        if photon_den <= 0:
            draws[i] = np.nan
            continue
        trial_scale = flux / photon_den
        trial_flux = float(_trapz(trial[0, rest] * energy[rest], energy[rest]) * KEV_TO_ERG) * trial_scale
        draws[i] = 4.0 * np.pi * distance ** 2 * trial_flux
    sample = draws[np.isfinite(draws)]
    if len(sample) < 100:
        raise RuntimeError(f"Only {len(sample)} valid peak-luminosity draws.")
    low, high = np.percentile(sample, [16, 84])
    ep_z = float(row["xp"] * (1.0 + redshift))
    return {
        "Lpeak": (float(luminosity), max(float(luminosity - low), 0.0), max(float(high - luminosity), 0.0)),
        "Ep_peak": (
            ep_z,
            float(row["xp_err_lo"] * (1.0 + redshift)),
            float(row["xp_err_hi"] * (1.0 + redshift)),
        ),
        "t_peak": peak_time,
        "t_start": float(row["t_start"]),
        "t_end": float(row["t_end"]),
        "scale": float(scale),
    }


def yonetoku_frame(sample, catalog, pflx):
    """Peak luminosity and peak-spectrum E_p,z for redshift-known GBM bursts."""
    pflx = pflx.copy()
    for column in pflx.columns:
        if column not in ("name", "trigger_name"):
            pflx[column] = pd.to_numeric(pflx[column], errors="coerce")
    pflx["trigger_norm"] = pflx["trigger_name"].map(_norm_name)
    spectra = pflx.drop_duplicates("trigger_norm").set_index("trigger_norm")
    energy = np.logspace(np.log10(0.2), np.log10(2.0e4), 3000)
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM0)
    rows = []
    for burst in sample.itertuples(index=False):
        matched = match_burst(catalog, burst.GRB)
        if matched is None or matched["trigger_norm"] not in spectra.index:
            continue
        shape = _peak_shape(spectra.loc[matched["trigger_norm"]])
        if shape is None or not np.isfinite(burst.z) or burst.z <= 0:
            continue
        alpha, epeak, beta, flux = shape
        distance = cosmo.luminosity_distance(float(burst.z)).to(u.cm).value
        luminosity = luminosity_from_peak_flux(alpha, epeak, beta, flux, float(burst.z), distance, energy)
        if not np.isfinite(luminosity) or luminosity <= 0:
            continue
        rows.append({
            "GRB": burst.GRB,
            "Ep": epeak * (1.0 + float(burst.z)),
            "Ep_err": 0.0,
            "Lpeak": luminosity,
            "class": burst_class(getattr(burst, "type", "")),
        })
    return pd.DataFrame(rows)


def write_parameter_table(path, values, info, rank, n_sample):
    rows = []
    notes = {
        "z": "Redshift used for the luminosity distance and the rest-frame peak energy.",
        "T90": "GBM bcat duration, 50-300 keV, the same quantity as FERMIGBRST t90.",
        "T05": (
            "Time between the 47.5% and 52.5% points of the 50-300 keV cumulative "
            f"photon fluence, from {info['T05_start']:.2f} s to {info['T05_stop']:.2f} s. "
            f"The same curve gives T90 = {info['T90_from_curve']:.2f} s "
            f"(catalog {values['T90'][0]:.2f} s) and T50 = {info['T50_from_curve']:.2f} s "
            f"(catalog {info['T50_catalog']:.2f} s). "
            "FERMIGBRST and the high-energy compilation do not tabulate T05."
        ),
        "Eiso": (
            f"Rest-frame 10 keV-10 MeV, H0={H0}, Om={OM0}. "
            f"Model 10-1000 keV fluence {info['s10_model']:.3e} erg/cm^2 compared with "
            f"bcat fluence {info['s10_bcat']:.3e} erg/cm^2. "
            f"Rank {rank} of {n_sample} by Eiso, counting this burst."
        ),
        "Ep": "Rest-frame nu F_nu peak of the time-integrated joint Band model.",
        "fluence": "GBM bcat energy fluence, 10-1000 keV.",
        "peak_flux": "GBM bcat peak photon flux on 1024 ms, 10-1000 keV.",
        "Epeak": "Observer-frame nu F_nu peak of the time-integrated joint Band model.",
        "alpha": "Fluence-weighted mean of the time-resolved Band alpha, weighted by 10-1000 keV fluence.",
        "beta": "Fluence-weighted mean of the time-resolved Band beta, weighted by 10-1000 keV fluence.",
        "Lpeak": (
            "Isotropic luminosity at the 1024 ms peak, rest-frame 10 keV-10 MeV. "
            f"The peak is at {info['t_peak']:.1f} s, inside the joint Band interval "
            f"{info['peak_interval_start']:.1f}-{info['peak_interval_end']:.1f} s, "
            f"and is {info['peak_scale']:.2f} times that interval's mean flux. "
            f"H0={H0}, Om={OM0}."
        ),
        "Ep_peak": "Rest-frame nu F_nu peak of the joint Band interval that contains the 1024 ms peak.",
    }
    for name, (center, lo, hi) in values.items():
        rows.append({
            "attribute": name,
            "value": center,
            "err_lo": lo,
            "err_hi": hi,
            "note": notes[name],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _ordinal(n):
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suffix}"


def _finite_sample(values, positive):
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    if positive:
        keep &= values > 0
    return values[keep]


def _log_edges(sample, value, max_bins=40):
    """Equal bins in log10, wide enough to contain the comparison sample and this burst."""
    pool = np.concatenate([np.asarray(sample, float), [float(value)]])
    pool = pool[np.isfinite(pool) & (pool > 0)]
    logv = np.log10(pool)
    iqr = float(np.subtract(*np.percentile(logv, [75, 25])))
    width = 2.0 * iqr / len(pool) ** (1.0 / 3.0) if iqr > 0 else 0.15
    span = float(logv.max() - logv.min())
    n_bins = int(np.clip(np.ceil(span / width), 16, max_bins))
    pad = 0.03 * span
    return np.logspace(logv.min() - pad, logv.max() + pad, n_bins + 1)


def _linear_edges(sample, value, max_bins=32):
    pool = np.concatenate([np.asarray(sample, float), [float(value)]])
    pool = pool[np.isfinite(pool)]
    iqr = float(np.subtract(*np.percentile(pool, [75, 25])))
    span = float(pool.max() - pool.min())
    width = 2.0 * iqr / len(pool) ** (1.0 / 3.0) if iqr > 0 else span / 20.0
    n_bins = int(np.clip(np.ceil(span / max(width, 1e-9)), 12, max_bins))
    pad = 0.04 * span if span > 0 else 0.5
    return np.linspace(pool.min() - pad, pool.max() + pad, n_bins + 1)


def _redshift_edges(sample, value):
    """Half-unit bins from zero through the highest redshift."""
    top = max(float(np.nanmax(sample)), float(value))
    return np.arange(0.0, np.floor(top) + 1.5, 0.5)


def _rank_against(sample, value):
    sample = np.asarray(sample, float)
    n = int(sample.size)
    n_above = int(np.sum(sample > value))
    return {
        "n": n,
        "n_above": n_above,
        "percentile": 100.0 * float(np.sum(sample < value)) / n,
        "rank": n_above + 1,
        "n_with_this": n + 1,
        "median": float(np.median(sample)),
    }


def _sci(value, digits=2):
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def _hist_value_text(key, value):
    if key == "z":
        return f"$z = {value:.2f}$"
    if key == "T90":
        return rf"$T_{{90}} = {value:.0f}\,\mathrm{{s}}$"
    if key == "fluence":
        return rf"${_sci(value)}\,\mathrm{{erg\,cm^{{-2}}}}$"
    if key == "peak_flux":
        return rf"${value:.1f}\,\mathrm{{ph\,cm^{{-2}}\,s^{{-1}}}}$"
    if key == "Epeak":
        return rf"$E_{{\mathrm{{peak}}}} = {value:.0f}\,\mathrm{{keV}}$"
    if key == "alpha":
        return rf"$\alpha = {value:.2f}$"
    if key == "beta":
        return rf"$\beta = {value:.2f}$"
    if key == "Eiso":
        return rf"$E_{{\mathrm{{iso}}}} = {_sci(value)}\,\mathrm{{erg}}$"
    if key == "Ep":
        return rf"$E_{{\mathrm{{p,z}}}} = {value:.0f}\,\mathrm{{keV}}$"
    if key == "Lpeak":
        return rf"$L_{{\mathrm{{peak}}}} = {_sci(value)}\,\mathrm{{erg\,s^{{-1}}}}$"
    return f"{value:.3g}"


def _draw_histogram(ax, sample, value, xlabel, value_text, star_label, log=False, edges=None):
    """Histogram of one property with the median and the target burst marked."""
    sample = _finite_sample(sample, positive=log or edges is not None)
    if edges is None:
        edges = _log_edges(sample, value) if log else _linear_edges(sample, value)
    counts, _, _ = ax.hist(
        sample, bins=edges, color=POP_COLOR, edgecolor=POP_EDGE, linewidth=0.35, zorder=2,
    )
    if log:
        ax.set_xscale("log")
    rank = _rank_against(sample, value)
    ax.axvline(rank["median"], color="#4b5563", lw=0.9, ls="--", zorder=3)
    ax.axvline(value, color="black", lw=1.15, zorder=4)
    ymax = max(float(counts.max()), 1.0)
    ax.set_ylim(0.0, ymax * 1.34)
    ax.plot([value], [ymax * 1.14], marker="*", ms=13, color="black", linestyle="none", zorder=5)

    if log and edges[0] > 0:
        position = (np.log10(value) - np.log10(edges[0])) / (np.log10(edges[-1]) - np.log10(edges[0]))
    else:
        position = (value - edges[0]) / (edges[-1] - edges[0])
    # Keep the label on the side away from the burst, where the axis is empty.
    on_left = position >= 0.5
    if rank["percentile"] >= 95.0 or rank["n_above"] <= 25:
        place = f"{_ordinal(rank['rank'])} of {rank['n_with_this']:,}\nabove {rank['percentile']:.1f}%"
    else:
        place = f"{_ordinal(int(round(rank['percentile'])))} percentile\n{rank['n_above']:,} of {rank['n']:,} higher"
    ax.text(
        0.04 if on_left else 0.96, 0.96,
        f"{star_label}\n{place}\n{value_text}",
        transform=ax.transAxes, ha="left" if on_left else "right", va="top", fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.94},
        zorder=6,
    )
    ax.set_xlabel(xlabel)
    _style_axes(ax)
    return rank


def _histogram_legend(fig, star_label):
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=POP_COLOR, edgecolor=POP_EDGE, linewidth=0.4, label="Long bursts"),
        plt.Line2D([0], [0], color="#4b5563", lw=0.9, ls="--", label="Median"),
        plt.Line2D([0], [0], color="black", lw=1.15, marker="*", ms=9, label=star_label),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0.955),
    )


def _apply_plot_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6,
    })


def plot_property_histograms(output_dir, values, grb_name, high_energy_csv=HIGH_ENERGY_CSV):
    """Histograms of the long-burst population with the target burst marked.

    Observed quantities use the Fermi-GBM catalog and the same T90 >= 2 s cut
    as the scatter plots. Redshift, isotropic energy, and rest-frame peak
    energy use the long bursts in the high-energy compilation. Peak luminosity
    uses the subset of those bursts with a GBM peak-flux spectrum.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    sample = pd.read_csv(high_energy_csv)
    sample = sample[sample["GRB"].map(_norm_name) != _norm_name(grb_name)].copy()
    sample["class"] = sample["type"].map(burst_class)
    long = sample[sample["class"] == "long"]
    catalog = gbm_science_frame(download_gbm_catalog(CACHE))
    catalog = catalog[~_burst_rows(catalog, grb_name)].copy()
    star_label = f"GRB {_norm_name(grb_name)}"
    catalog["class"] = catalog["T90"].map(duration_class)
    gbm_long = catalog[catalog["class"] == "long"]
    luminosities = yonetoku_frame(sample, catalog, download_pflx_catalog(PFLX_CACHE))
    luminous = luminosities[luminosities["class"] == "long"]

    observed = (
        ("T90", gbm_long["T90"], True, None, r"$T_{90}$ (s)"),
        ("fluence", gbm_long["fluence"], True, None, r"Fluence, 10–1000 keV (erg cm$^{-2}$)"),
        ("peak_flux", gbm_long["peak_flux"], True, None, r"Peak flux, 1024 ms (ph cm$^{-2}$ s$^{-1}$)"),
        ("Epeak", gbm_long["Epeak"], True, None, r"$E_{\mathrm{peak}}$ (keV)"),
        ("alpha", gbm_long["alpha"], False, None, r"$\alpha$"),
        ("beta", gbm_long["beta"], False, None, r"$\beta$"),
    )
    eiso = long["Eiso"].to_numpy(float) * 1.0e52
    inferred = (
        ("Eiso", eiso, True, None, r"$E_{\mathrm{iso}}$ (erg)"),
        ("Ep", long["Ep"], True, None, r"$E_{\mathrm{p,z}}$ (keV)"),
        ("Lpeak", luminous["Lpeak"], True, None, r"$L_{\mathrm{peak}}$ (erg s$^{-1}$), GBM peak spectrum"),
        ("z", long["z"], False, "redshift", r"Redshift $z$"),
    )

    def draw_figure(path, panels, ncols, suptitle, figsize):
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ranks = []
        for index, (key, series, log, edges_key, xlabel) in enumerate(panels):
            ax = axes[index // ncols][index % ncols]
            value = float(values[key][0])
            if key == "Eiso":
                value *= 1.0e52
            edges = _redshift_edges(series, value) if edges_key == "redshift" else None
            rank = _draw_histogram(
                ax, series, value, xlabel, _hist_value_text(key, value), star_label,
                log=log, edges=edges,
            )
            rank["property"] = key
            ranks.append(rank)
            if index % ncols == 0:
                ax.set_ylabel("Number of bursts")
        _histogram_legend(fig, star_label)
        fig.suptitle(suptitle, fontsize=12, y=0.995)
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.86, hspace=0.42, wspace=0.28)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")
        return ranks

    observed_ranks = draw_figure(
        output_dir / "hist_observed.png", observed, 3,
        r"Observed properties of long Fermi-GBM bursts ($T_{90} \geq 2$ s)",
        (11.4, 7.0),
    )
    inferred_ranks = draw_figure(
        output_dir / "hist_inferred.png", inferred, 2,
        "Inferred properties of redshift-known long bursts",
        (9.8, 7.0),
    )
    rows = []
    for kind, ranks in (("observed", observed_ranks), ("inferred", inferred_ranks)):
        for rank in ranks:
            rows.append({
                "property": rank["property"],
                "kind": kind,
                "n_comparison": rank["n"],
                "value": values[rank["property"]][0],
                "median": rank["median"] / 1.0e52 if rank["property"] == "Eiso" else rank["median"],
                "percentile": rank["percentile"],
                "rank": rank["rank"],
                "n_with_this": rank["n_with_this"],
                "n_higher": rank["n_above"],
            })
    table = pd.DataFrame(rows)
    table_path = output_dir / "histogram_ranks.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")
    print(table.to_string(index=False, float_format=lambda item: f"{item:.4g}"))
    return table


def _log_errors(value, err):
    """Asymmetric errors for a logarithmic axis.

    The lower bar is omitted when the printed uncertainty reaches zero, so one
    unconstrained point does not stretch the axis by orders of magnitude.
    """
    value = np.asarray(value, float)
    err = np.asarray(err, float)
    err = np.where(np.isfinite(err) & (err > 0), err, 0.0)
    lower = np.where(value > err, err, 0.0)
    return [lower, err]


def plot_frail_relation(output_dir, eiso_1e52, eiso_err_lo, eiso_err_hi, theta_deg, grb_label):
    """E_iso versus jet angle for Frail et al. (2001) and Wang et al. (2018).

    Frail et al. (2001) Table 1 gives the pre-Swift angles, in radians in that
    file. Wang et al. (2018) Table 5 gives the jet-break sample, with angles
    already in degrees. A power law is fit to every measured angle in the two
    tables. Frail limits are drawn and left out of the fit. Bursts that appear
    in both papers are drawn twice, once for each measurement. The target burst
    is placed at its isotropic energy and the supplied jet angle.
    """
    frail = pd.read_csv(FRAIL_TABLE, comment="#")
    measured = frail[frail["theta_j"].notna() & frail["theta_j_limit"].fillna("").isin(["", "~"])].copy()
    lower = frail[frail["theta_j_limit"] == ">"].copy()
    upper = frail[frail["theta_j_limit"] == "<"].copy()
    wang = pd.read_csv(WANG_TABLE, comment="#")
    wang = wang[wang["theta_j"].notna() & wang["E_gamma_iso"].notna()].copy()

    def degrees(frame):
        return np.degrees(frame["theta_j"].to_numpy(float))

    def erg(frame):
        return frame["E_iso"].to_numpy(float) * 1.0e51

    frail_theta = degrees(measured)
    frail_energy = erg(measured)
    wang_theta = wang["theta_j"].to_numpy(float)
    wang_energy = wang["E_gamma_iso"].to_numpy(float) * 1.0e52
    wang_theta_err = _log_errors(wang_theta, wang["theta_j_err"].to_numpy(float))
    wang_energy_err = _log_errors(wang_energy, wang["E_gamma_iso_err"].to_numpy(float) * 1.0e52)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.scatter(
        frail_theta, frail_energy, s=36, c=POP_COLOR, edgecolors=POP_EDGE,
        linewidths=0.4, zorder=3, label=f"Frail et al. (2001) ({len(measured)})",
    )
    for frame, direction in ((lower, 1), (upper, -1)):
        for theta_value, energy in zip(degrees(frame), erg(frame)):
            ax.annotate(
                "",
                xy=(theta_value * 10 ** (0.08 * direction), energy),
                xytext=(theta_value, energy),
                arrowprops={"arrowstyle": "->", "color": POP_EDGE, "lw": 0.8},
            )
            ax.scatter([theta_value], [energy], s=36, facecolors="none", edgecolors=POP_EDGE, linewidths=0.9, zorder=3)
    if len(lower) or len(upper):
        ax.scatter([], [], s=36, facecolors="none", edgecolors=POP_EDGE, linewidths=0.9, label="Limit on " + r"$\theta_j$")

    ax.errorbar(
        wang_theta, wang_energy, xerr=wang_theta_err, yerr=wang_energy_err,
        fmt="none", ecolor=WANG_EDGE, elinewidth=0.55, capsize=0, alpha=0.55, zorder=2,
    )
    ax.scatter(
        wang_theta, wang_energy, s=28, marker="D", c=WANG_COLOR, edgecolors=WANG_EDGE,
        linewidths=0.35, zorder=4, label=f"Wang et al. (2018) ({len(wang)})",
    )

    eiso = eiso_1e52 * 1.0e52
    ax.errorbar(
        [theta_deg], [eiso], yerr=[[eiso_err_lo * 1.0e52], [eiso_err_hi * 1.0e52]],
        fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=5,
    )
    ax.scatter([theta_deg], [eiso], s=220, marker="*", c="black", zorder=6, label=grb_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    add_relation_fit(
        ax,
        np.concatenate([frail_theta, wang_theta]),
        np.concatenate([frail_energy, wang_energy]),
        True, True,
    )
    ax.set_xlabel(r"Jet opening angle $\theta_j$ (deg)")
    ax.set_ylabel(r"$E_{\mathrm{iso}}$ (erg)")
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.94, edgecolor="none", loc="center left", bbox_to_anchor=(1.02, 0.5))
    outfile = Path(output_dir) / "frail_relation.png"
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Jet angles: {len(measured)} from Frail et al. (2001), "
        f"{len(lower) + len(upper)} Frail limits, {len(wang)} from Wang et al. (2018) Table 5"
    )
    print(f"Wrote {outfile}")
    return outfile


def _beaming_factor(theta_deg):
    """Fraction of the isotropic energy inside a jet of half-opening angle theta_deg."""
    return 1.0 - np.cos(np.deg2rad(np.asarray(theta_deg, float)))


def plot_ghirlanda_relation(
    output_dir, eiso_1e52, eiso_err_lo, eiso_err_hi, ep_z, ep_err_lo, ep_err_hi, theta_deg, grb_label,
):
    """Rest-frame peak energy versus collimation-corrected energy.

    This is the Ghirlanda relation. Wang et al. (2018) Table 5 supplies
    E_gamma, and Table 2 of that paper supplies the redshift and the
    observer-frame peak, with E_p,z = (1+z) E_p. Frail et al. (2001) Table 1
    has no peak energy, so those bursts are not drawn. A power law is fit to
    the Wang points. The target burst uses E_gamma = E_iso (1 - cos theta_j).
    """
    wang = pd.read_csv(WANG_TABLE, comment="#", dtype={"GRB": str})
    keep = wang["E_gamma"].notna() & wang["Ep"].notna() & wang["z"].notna()
    keep &= (wang["E_gamma"] > 0) & (wang["Ep"] > 0) & (wang["z"] > -1)
    wang = wang[keep].copy()
    energy = wang["E_gamma"].to_numpy(float) * 1.0e51
    energy_err = _log_errors(energy, wang["E_gamma_err"].to_numpy(float) * 1.0e51)
    redshift = 1.0 + wang["z"].to_numpy(float)
    peak = wang["Ep"].to_numpy(float) * redshift
    peak_err = _log_errors(peak, wang["Ep_err"].to_numpy(float) * redshift)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.errorbar(
        energy, peak, xerr=energy_err, yerr=peak_err,
        fmt="none", ecolor=WANG_EDGE, elinewidth=0.55, capsize=0, alpha=0.55, zorder=2,
    )
    ax.scatter(
        energy, peak, s=28, marker="D", c=WANG_COLOR, edgecolors=WANG_EDGE,
        linewidths=0.35, zorder=4, label=f"Wang et al. (2018) ({len(wang)})",
    )

    beaming = float(_beaming_factor(theta_deg))
    this_energy = eiso_1e52 * 1.0e52 * beaming
    ax.errorbar(
        [this_energy], [ep_z],
        xerr=[[eiso_err_lo * 1.0e52 * beaming], [eiso_err_hi * 1.0e52 * beaming]],
        yerr=[[ep_err_lo], [ep_err_hi]],
        fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=5,
    )
    ax.scatter([this_energy], [ep_z], s=220, marker="*", c="black", zorder=6, label=grb_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    add_relation_fit(ax, energy, peak, True, True)
    ax.set_xlabel(r"$E_{\gamma}$ (erg)")
    ax.set_ylabel(r"$E_{\mathrm{p,z}}$ (keV)")
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.94, edgecolor="none", loc="center left", bbox_to_anchor=(1.02, 0.5))
    outfile = Path(output_dir) / "ghirlanda_relation.png"
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Ghirlanda relation: {len(wang)} jet-break measurements from Wang et al. (2018)")
    print(f"Wrote {outfile}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grb", default=DEFAULT_GRB, help="GRB name, for example 260924A or 221009A. Default: 260924A.")
    parser.add_argument("--redshift", type=float, default=None, help="Redshift. Taken from the high-energy compilation when omitted.")
    parser.add_argument("--bcat", type=Path, default=None, help="GBM bcat FITS file. Default: data/GRB<name>/gbm/glg_bcat_all_*.fit.")
    parser.add_argument("--band-csv", type=Path, default=None, help="Time-resolved Band fits. Default: results/GRB<name>/band_joint_gbm_bat_xrt.csv.")
    parser.add_argument("--theta", type=float, default=None, help="Jet half-opening angle in degrees. Omit to skip the Frail and Ghirlanda panels.")
    parser.add_argument("--high-energy-csv", type=Path, default=HIGH_ENERGY_CSV)
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure directory. Default: results/GRB<name>/population.")
    parser.add_argument("--refresh-catalog", action="store_true", help="Download the GBM catalog again instead of using the cache.")
    parser.add_argument("--draws", type=int, default=800, help="Monte Carlo draws for the derived uncertainties.")
    parser.add_argument(
        "--histograms-only", action="store_true",
        help="Write the population histograms from the parameter table and skip the scatter plots.",
    )
    args = parser.parse_args()
    target = resolve_target(args)

    if args.histograms_only:
        table = pd.read_csv(target.output_dir / target.parameter_csv)
        values = {
            row.attribute: (float(row.value), float(row.err_lo), float(row.err_hi))
            for row in table.itertuples(index=False)
        }
        plot_property_histograms(target.output_dir, values, target.name, args.high_energy_csv)
        return

    if target.bcat is None or not target.bcat.is_file():
        raise SystemExit(
            f"No GBM bcat for GRB {target.name} under data/GRB{target.name}/gbm/. Pass --bcat."
        )
    if not target.band.is_file():
        raise SystemExit(
            f"No Band table at {target.band}. Pass --band-csv."
        )

    sample = pd.read_csv(args.high_energy_csv)
    sample["GRB"] = sample["GRB"].astype(str).str.strip()
    redshift = resolve_redshift(args.redshift, target.name, sample)
    sample = sample[sample["GRB"].map(_norm_name) != target.name].copy()
    top = top_energy_bursts(sample, N_TOP)
    print("Highest isotropic energies in the high-energy sample:")
    print(top[["GRB", "z", "T90", "Eiso", "Ep"]].to_string(index=False))

    values, info = this_grb_parameters(target.bcat, target.band, redshift, n_draws=args.draws, seed=target.seed)
    peak = this_burst_peak_luminosity(target.bcat, target.band, redshift, n_draws=args.draws, seed=target.seed)
    values["Lpeak"] = peak["Lpeak"]
    values["Ep_peak"] = peak["Ep_peak"]
    info["t_peak"] = peak["t_peak"]
    info["peak_interval_start"] = peak["t_start"]
    info["peak_interval_end"] = peak["t_end"]
    info["peak_scale"] = peak["scale"]
    n_with_this = len(sample) + 1
    n_brighter = int((sample["Eiso"] > values["Eiso"][0]).sum())
    rank = n_brighter + 1
    print(
        f"{target.label}: z={values['z'][0]:.2f}, "
        f"T90={values['T90'][0]:.1f} s, "
        f"T05={values['T05'][0]:.2f} s "
        f"(-{values['T05'][1]:.2f}/+{values['T05'][2]:.2f}, "
        f"{info['T05_start']:.2f}–{info['T05_stop']:.2f} s), "
        f"Eiso={values['Eiso'][0]:.1f} x 10^52 erg "
        f"(-{values['Eiso'][1]:.1f}/+{values['Eiso'][2]:.1f}), "
        f"Ep,z={values['Ep'][0]:.0f} keV "
        f"(-{values['Ep'][1]:.0f}/+{values['Ep'][2]:.0f}), "
        f"Lpeak={values['Lpeak'][0]:.2e} erg/s "
        f"(-{values['Lpeak'][1]:.2e}/+{values['Lpeak'][2]:.2e}) "
        f"at Ep,z={values['Ep_peak'][0]:.0f} keV"
    )
    print(
        f"Rank by Eiso: {rank} of {n_with_this}. "
        f"Joint-fit 10-1000 keV fluence is {info['s10_model'] / info['s10_bcat']:.2f} times the GBM bcat fluence "
        f"({info['n_valid']}/{info['n_draws']} Monte Carlo draws kept)."
    )

    catalog = gbm_science_frame(download_gbm_catalog(CACHE, refresh=args.refresh_catalog))
    background = catalog[~_burst_rows(catalog, target.name)].copy()

    target.output_dir.mkdir(parents=True, exist_ok=True)
    write_parameter_table(target.output_dir / target.parameter_csv, values, info, rank, n_with_this)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6,
    })

    high_highlights = []
    for _, row in top.iterrows():
        high_highlights.append({
            "label": f"GRB {row['GRB']}",
            "values": high_energy_record(row),
            "class": burst_class(row["type"]),
        })
    high_background = sample[~sample["GRB"].isin(top["GRB"])].copy()
    high_background["class"] = high_background["type"].map(burst_class)
    # The plotting helper looks up T90_err optionally and z, Eiso, Ep by name.
    for x_key, y_key in itertools.combinations(HIGH_ENERGY_ATTRS, 2):
        outfile = target.output_dir / f"high_energy_{x_key}_vs_{y_key}.png"
        plot_pair(outfile, x_key, y_key, high_background, high_highlights, values, "High-energy GRBs", target.label)
        print(f"Wrote {outfile}")

    gbm_highlights = []
    for _, row in top.iterrows():
        matched = match_burst(catalog, row["GRB"])
        if matched is None:
            print(f"GRB {row['GRB']} is not in the Fermi-GBM catalog; it is omitted from the GBM panels.")
            gbm_highlights.append({
                "label": f"GRB {row['GRB']}",
                "values": {key: (np.nan, np.nan, np.nan) for key in GBM_ATTRS},
                "class": burst_class(row["type"]),
            })
            continue
        print(f"GRB {row['GRB']} matched to GBM {matched['name']} ({matched['trigger_name']}).")
        record = gbm_record(matched)
        # A few of the brightest bursts have no converged fluence-spectrum fit in the
        # GBM catalog. The observer-frame peak is then the high-energy compilation's
        # rest-frame Ep divided by (1+z), which is the same peak used on the other panels.
        if not np.isfinite(record["Epeak"][0]) and np.isfinite(row["Ep"]) and float(row["z"]) > -0.999:
            scale = 1.0 + float(row["z"])
            record["Epeak"] = (float(row["Ep"]) / scale, float(row["Ep_err"]) / scale, float(row["Ep_err"]) / scale)
            print(f"  GBM has no fluence-spectrum peak for GRB {row['GRB']}; using Ep/(1+z) = {record['Epeak'][0]:.0f} keV.")
        gbm_highlights.append({
            "label": f"GRB {row['GRB']}",
            "values": record,
            "class": duration_class(matched["T90"]),
        })
        background = background[background["norm"] != matched["norm"]]
    background["class"] = background["T90"].map(duration_class)

    for x_key, y_key in itertools.combinations(GBM_ATTRS, 2):
        outfile = target.output_dir / f"gbm_{x_key}_vs_{y_key}.png"
        plot_pair(outfile, x_key, y_key, background, gbm_highlights, values, "Fermi-GBM", target.label)
        print(f"Wrote {outfile}")

    pflx = download_pflx_catalog(PFLX_CACHE, refresh=args.refresh_catalog)
    yonetoku = yonetoku_frame(sample, catalog, pflx)
    yonetoku_background = yonetoku[~yonetoku["GRB"].isin(top["GRB"])]
    yonetoku_highlights = []
    for _, burst in top.iterrows():
        matched = yonetoku[yonetoku["GRB"] == burst["GRB"]]
        if matched.empty:
            print(f"GRB {burst['GRB']} has no GBM peak-flux spectrum; it is omitted from the Lpeak panel.")
            yonetoku_highlights.append({
                "label": f"GRB {burst['GRB']}",
                "values": {"Ep": (np.nan, np.nan, np.nan), "Lpeak": (np.nan, np.nan, np.nan)},
                "class": burst_class(burst["type"]),
            })
            continue
        point = matched.iloc[0]
        yonetoku_highlights.append({
            "label": f"GRB {burst['GRB']}",
            "values": {
                "Ep": (float(point["Ep"]), 0.0, 0.0),
                "Lpeak": (float(point["Lpeak"]), 0.0, 0.0),
            },
            "class": str(point["class"]),
        })
    this_yonetoku = dict(values)
    this_yonetoku["Ep"] = values["Ep_peak"]
    outfile = target.output_dir / "high_energy_Lpeak_vs_Ep.png"
    plot_pair(outfile, "Ep", "Lpeak", yonetoku_background, yonetoku_highlights, this_yonetoku, "Redshift-known GBM GRBs", target.label)
    print(f"Wrote {outfile}")
    if target.theta is None:
        print(f"No jet angle for {target.label}. Pass --theta to draw the Frail and Ghirlanda panels.")
    else:
        plot_frail_relation(
            target.output_dir, values["Eiso"][0], values["Eiso"][1], values["Eiso"][2], target.theta, target.label,
        )
        plot_ghirlanda_relation(
            target.output_dir,
            values["Eiso"][0], values["Eiso"][1], values["Eiso"][2],
            values["Ep"][0], values["Ep"][1], values["Ep"][2],
            target.theta, target.label,
        )
    plot_property_histograms(target.output_dir, values, target.name, args.high_energy_csv)


if __name__ == "__main__":
    main()
