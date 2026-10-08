"""Afterglow model (jetsimpy or VegasAfterglow) and the arrays the plots consume.

Choose the library with :func:`set_afterglow_library` or the ``--library``
command-line flag (``jetsimpy`` or ``vegas``). Plotting reads ``plot_data.pkl``
written by :func:`save_plot_data` and does not evaluate the model.
"""

import copy
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo

logger = logging.getLogger(__name__)

# FluxDensity / F_ν: erg/s/cm^2/Hz per mJy. jetsimpy Flux is band-integrated cgs.
_MJY_PER_CGS_FNU = 1e-26

# jetsimpy wind: n = nwind * (r / 1e17 cm)^{-2}. Vegas A_star=1 has
# n = 5e11 / r^2, so A_star = nwind * (1e17)^2 / 5e11.
_VEGAS_A_STAR_PER_NWIND = 3.0e-3  # (1e17) ** 2 / 5e11

_LIBRARY_ALIASES = {
    "jetsimpy": "jetsimpy",
    "vegas": "vegas",
    "vegasafterglow": "vegas",
}
_afterglow_library = "jetsimpy"

PLOT_DATA_FILENAME = "plot_data.pkl"

# Observer times (s) used when overlaying spectrum observations from a photometry CSV.
SPECTRUM_PLOT_TIME_EPOCHS = np.array(
    [
        1e3,
        4e3,
        1e4,
        15934,
        72000,
        190000,
        225000,
        1e6,
    ],
    dtype=float,
)


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
        df["UL"] = df["UL"].fillna("N")
        df["UL"] = df["UL"].astype(str).str.strip()
        df.loc[df["UL"] == "", "UL"] = "N"
    return df


def _filt_freqs_from_obs(df):
    """Map each filter label to its frequency (Hz) from the observation table."""
    if "Filt" not in df.columns or "Freqs" not in df.columns:
        raise ValueError(
            "observed data must contain Filt and Freqs columns to set plot frequencies"
        )
    work = df[["Filt", "Freqs"]].copy()
    work["Filt"] = work["Filt"].astype(str).str.strip()
    work["Freqs"] = pd.to_numeric(work["Freqs"], errors="coerce")
    work = work[(work["Filt"] != "") & (work["Filt"] != "nan")]
    work = work.dropna(subset=["Freqs"])
    filt_freqs = {}
    for band, group in work.groupby("Filt", sort=False):
        freqs = np.unique(group["Freqs"].to_numpy())
        if len(freqs) > 1:
            logger.warning(
                "Filter %r has multiple frequencies %s; using the median.",
                band,
                freqs.tolist(),
            )
        filt_freqs[band] = float(np.median(freqs))
    if not filt_freqs:
        raise ValueError("observed data has no usable Filt/Freqs rows")
    return filt_freqs


def resolve_afterglow_library(library=None):
    """Return ``jetsimpy`` or ``vegas``. ``None`` uses the current selection."""
    name = _afterglow_library if library is None else str(library).strip().lower()
    try:
        return _LIBRARY_ALIASES[name]
    except KeyError:
        known = ", ".join(sorted({"jetsimpy", "vegas"}))
        raise ValueError(f"unknown afterglow library {library!r}; choose {known}")


def set_afterglow_library(library):
    """Select the library used when ``model`` is called without ``library``."""
    global _afterglow_library
    _afterglow_library = resolve_afterglow_library(library)
    return _afterglow_library


def add_afterglow_library_argument(parser):
    """Add ``--library {jetsimpy,vegas}`` to an argparse parser."""
    parser.add_argument(
        "--library",
        type=str,
        default="jetsimpy",
        choices=["jetsimpy", "vegas"],
        help="Afterglow library: jetsimpy or vegas (VegasAfterglow).",
    )


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
    return _jetsimpy_jet_and_P(params)


def _jetsimpy_jet_and_P(params):
    """Build a jetsimpy jet from already-expanded physical parameters."""
    import jetsimpy

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


def _vegas_medium(n0, nwind):
    """Map jetsimpy ``nism`` + ``nwind`` onto a VegasAfterglow medium."""
    from VegasAfterglow import ISM, Wind

    if nwind == 0:
        return ISM(n0)
    A_star = nwind  # * _VEGAS_A_STAR_PER_NWIND
    if n0 == 0:
        return Wind(A_star, k_m=2.0)
    return Wind(A_star, n_ism=n0, k_m=2.0)


def _vegas_jet(params):
    """Map jetType / thc / e0 / lf / s onto a spreading VegasAfterglow jet."""
    from VegasAfterglow import GaussianJet, PowerLawJet, TophatJet

    theta_c = params["thc"]
    E_iso = params["e0"]
    Gamma0 = params["lf"]
    jet_type = str(params["jetType"]).lower()
    if jet_type == "gaussian":
        return GaussianJet(theta_c, E_iso, Gamma0, spreading=False)
    if jet_type == "powerlaw":
        s = params["s"]
        return PowerLawJet(theta_c, E_iso, Gamma0, s, s, spreading=False)
    return TophatJet(theta_c, E_iso, Gamma0, spreading=False)


def _build_vegas_model(params):
    """VegasAfterglow model. ``params`` must already be expanded to physical keys."""
    from VegasAfterglow import Model, Observer, Radiation

    dl_cm = cosmo.luminosity_distance(params["z"]).to("cm").value
    return Model(
        jet=_vegas_jet(params),
        medium=_vegas_medium(params["n0"], params["A"]),
        observer=Observer(lumi_dist=dl_cm, z=params["z"], theta_obs=params["thv"]),
        fwd_rad=Radiation(eps_e=params["epse"], eps_B=params["epsb"], p=params["p"]),
    )


def _broadcast_time_frequency(obs_time, obs_nu):
    """Match jetsimpy's vectorized (t, nu) broadcasting. Returns flat arrays."""
    times = np.asarray(obs_time, dtype=float)
    nus = np.asarray(obs_nu, dtype=float)
    scalar = times.ndim == 0 and nus.ndim == 0
    times = np.atleast_1d(times)
    nus = np.atleast_1d(nus)
    if times.size == 1 and nus.size != 1:
        times = np.full(nus.shape, times.reshape(-1)[0])
    elif nus.size == 1 and times.size != 1:
        nus = np.full(times.shape, nus.reshape(-1)[0])
    elif times.shape != nus.shape:
        raise ValueError(
            f"time and frequency shapes {times.shape} and {nus.shape} cannot be broadcast"
        )
    return times.reshape(-1), nus.reshape(-1), scalar, times.shape


def _vegas_flux_density_mjy(obs_time, obs_nu, params):
    """Synchrotron F_ν in mJy. ``params`` must already be expanded."""
    # logger.info(f"_vegas_flux_density_mjy params: {params}")
    times, nus, scalar, shape = _broadcast_time_frequency(obs_time, obs_nu)
    vegas_model = _build_vegas_model(params)
    order = np.argsort(times, kind="mergesort")
    if np.unique(nus).size == 1:
        grid = vegas_model.flux_density_grid(times[order], np.array([nus[0]]))
        fnu_sorted = np.asarray(grid.total, dtype=float).reshape(-1)
    else:
        paired = vegas_model.flux_density(times[order], nus[order])
        fnu_sorted = np.asarray(paired.total, dtype=float).reshape(-1)
    fnu = np.empty(times.size, dtype=float)
    fnu[order] = fnu_sorted
    fnu = fnu / _MJY_PER_CGS_FNU
    if scalar:
        return fnu.reshape(())
    # logger.info(f"_vegas_flux_density_mjy returning: {fnu}")
    return fnu.reshape(shape)


def _jetsimpy_flux_density_mjy(obs_time, obs_nu, params):
    jet, P = _jetsimpy_jet_and_P(params)
    return jet.FluxDensity(
        obs_time,
        obs_nu,
        P,
        model="sync",
        rtol=1e-3,
        max_iter=100,
        force_return=True,
    )


def model(obs_time, obs_nu, params, library=None):
    """Synchrotron flux density in mJy at observer times (s) and frequencies (Hz)."""
    library = resolve_afterglow_library(library)
    p = copy.deepcopy(params)
    _expand_jetsimpy_params_inplace(p)
    if library == "vegas":
        return _vegas_flux_density_mjy(obs_time, obs_nu, p)
    return _jetsimpy_flux_density_mjy(obs_time, obs_nu, p)


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

    if n0 <= 0:
        n0 = 1.0  # Pure wind case, we set a canonical value for the ISM number density
        nu_c = (
            2.7e12
            * (1 + z) ** (-0.5)
            * eps_b ** (-1.5)
            * E52 ** (-0.5)
            * n0 ** (-1)
            * td ** (-0.5)
        )
    else:
        nu_c = np.nan
    return nu_m, nu_c


def compute_break_frequencies_from_spectrum(jet, P, times, n_freq_bins=160):
    """
    Infer ν_m and ν_c at each observer time from synchrotron spectra built with ``jet.Flux``.

    Frequency grid: 1e9–1e19 Hz in ``n_freq_bins`` logarithmic bins; bin flux is converted to
    F_ν by dividing by bin width. Breaks are estimated with ``_estimate_break_frequencies``.
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


def _spectrum_on_grid(jet, P, times, n_freq_bins):
    """F_ν (mJy) on a log frequency grid at each observer time."""
    nu_edges = np.geomspace(1e9, 1e19, n_freq_bins + 1)
    nu_lo = nu_edges[:-1]
    nu_hi = nu_edges[1:]
    dnu = nu_hi - nu_lo
    nu_c = np.sqrt(nu_lo * nu_hi)
    times = np.asarray(times, dtype=float)
    fnu = np.empty((len(times), len(nu_c)), dtype=float)
    for it, t_obs in enumerate(times):
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
            fnu[it, i] = fband / dnu[i] / _MJY_PER_CGS_FNU
    return nu_c, fnu


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


def _load_observations(observed_data):
    df = pd.read_csv(observed_data)
    df["Times"] = pd.to_numeric(df["Times"], errors="coerce")
    df["Fluxes"] = pd.to_numeric(df["Fluxes"], errors="coerce")
    df["FluxErrs"] = pd.to_numeric(df["FluxErrs"], errors="coerce")
    if "Freqs" in df.columns:
        df["Freqs"] = pd.to_numeric(df["Freqs"], errors="coerce")
    df = _normalize_ul_column(df)
    if "Filt" in df.columns:
        df["Filt"] = df["Filt"].astype(str).str.strip()
    return df


def _vegas_spectrum_on_grid(params, times, n_freq_bins):
    """F_ν (mJy) on a log frequency grid. ``params`` must already be expanded."""
    nu_edges = np.geomspace(1e9, 1e19, n_freq_bins + 1)
    nu_lo = nu_edges[:-1]
    nu_hi = nu_edges[1:]
    nu_c = np.sqrt(nu_lo * nu_hi)
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="mergesort")
    vegas_model = _build_vegas_model(params)
    grid = vegas_model.flux_density_grid(times[order], nu_c)
    fnu_sorted = np.asarray(grid.total, dtype=float) / _MJY_PER_CGS_FNU
    fnu = np.empty((len(times), len(nu_c)), dtype=float)
    fnu[order] = fnu_sorted.T
    return nu_c, fnu


def _model_flux_or_nan(times, nu, params, what, library=None):
    try:
        flux = np.asarray(
            model(times, [nu], params, library=library), dtype=float
        ).reshape(-1)
    except Exception as e:
        logger.error("model failed for %s: %s", what, e)
        return np.full(len(np.atleast_1d(times)), np.nan, dtype=float)
    if flux.size != len(np.atleast_1d(times)):
        logger.error(
            "model returned %d fluxes for %s; expected %d",
            flux.size,
            what,
            len(np.atleast_1d(times)),
        )
        return np.full(len(np.atleast_1d(times)), np.nan, dtype=float)
    return flux


def compute_plot_data(
    median_params,
    sample_params,
    observed_data,
    xlim,
    spectrum_epochs=None,
    spectrum_dt_sec=500.0,
    n_freq_bins_spectrum=100,
    n_freq_bins_breaks=160,
    n_break_times=60,
    flux_times_days=(1, 2, 4, 8, 16),
    library=None,
):
    """
    Evaluate the afterglow model once and pack every array the plots need.

    ``sample_params`` are the posterior draws drawn as faint light-curve traces.
    ``xlim`` is the observer-time window (seconds) of the light-curve grid.
    ``library`` selects jetsimpy or VegasAfterglow; the default is the value
    set by :func:`set_afterglow_library`.
    """
    library = resolve_afterglow_library(library)
    ta, tb = xlim
    lc_times = np.geomspace(ta, tb, num=100)
    df = _load_observations(observed_data)
    filt_freqs = _filt_freqs_from_obs(df)
    bands = [band for band, _nu in sorted(filt_freqs.items(), key=lambda x: -x[1])]
    sample_params = list(sample_params)

    precomputed = {}
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for band in bands:
            nu = filt_freqs[band]
            future = executor.submit(
                _model_flux_or_nan,
                lc_times,
                nu,
                median_params,
                f"{band} median",
                library,
            )
            future_map[future] = (band, "median")
            for i, params in enumerate(sample_params):
                future = executor.submit(
                    _model_flux_or_nan,
                    lc_times,
                    nu,
                    params,
                    f"{band} sample {i}",
                    library,
                )
                future_map[future] = (band, i)
        for fut in as_completed(future_map):
            band, tag = future_map[fut]
            precomputed[(band, tag)] = fut.result()

    light_curves = []
    for band in bands:
        sample_fluxes = (
            np.vstack([precomputed[(band, i)] for i in range(len(sample_params))])
            if sample_params
            else np.empty((0, len(lc_times)))
        )
        light_curves.append(
            {
                "band": band,
                "nu": filt_freqs[band],
                "median_flux": precomputed[(band, "median")],
                "sample_fluxes": sample_fluxes,
            }
        )

    checkpoints = []
    flux_times_days = list(flux_times_days)
    flux_times_seconds = [d * 86400.0 for d in flux_times_days]
    for band in bands:
        nu = filt_freqs[band]
        flux_values = _model_flux_or_nan(
            flux_times_seconds, nu, median_params, f"{band} checkpoints", library
        )
        for day, sec, flux in zip(flux_times_days, flux_times_seconds, flux_values):
            checkpoints.append(
                {
                    "band": band,
                    "nu": nu,
                    "time_days": day,
                    "time_seconds": sec,
                    "flux_mjy": float(flux),
                }
            )

    df["ModelFlux"] = np.nan
    for band in bands:
        mask = (df["Filt"] == band) & (df["UL"].astype(str) == "N")
        times = df.loc[mask, "Times"].to_numpy(dtype=float)
        if len(times) == 0:
            continue
        df.loc[mask, "ModelFlux"] = _model_flux_or_nan(
            times, filt_freqs[band], median_params, f"{band} residuals", library
        )

    epochs = (
        np.asarray(SPECTRUM_PLOT_TIME_EPOCHS, dtype=float)
        if spectrum_epochs is None
        else np.asarray(spectrum_epochs, dtype=float)
    )
    physical = copy.deepcopy(median_params)
    _expand_jetsimpy_params_inplace(physical)
    if library == "vegas":
        spectrum_nu, spectrum_fnu = _vegas_spectrum_on_grid(
            physical, epochs, n_freq_bins_spectrum
        )
        jet, P = None, None
    else:
        jet, P = _jetsimpy_jet_and_P(physical)
        spectrum_nu, spectrum_fnu = _spectrum_on_grid(
            jet, P, epochs, n_freq_bins_spectrum
        )
    epoch_observations = build_spectrum_epoch_observations(
        df, epochs, dt_sec=spectrum_dt_sec
    )

    break_times = np.geomspace(1.0, 1.0e6, num=n_break_times)
    if str(physical["jetType"]).lower() == "tophat":
        nu_m, nu_c = compute_break_frequencies_tophat_analytical(
            physical["z"],
            physical["p"],
            physical["epse"],
            physical["epsb"],
            physical["e0"],
            physical["n0"],
            break_times,
        )
        break_method = "analytical"
    elif library == "vegas":
        break_nu, break_fnu = _vegas_spectrum_on_grid(
            physical, break_times, n_freq_bins_breaks
        )
        nu_m = np.empty(len(break_times), dtype=float)
        nu_c = np.empty(len(break_times), dtype=float)
        for it in range(len(break_times)):
            nu_m[it], nu_c[it] = _estimate_break_frequencies(break_nu, break_fnu[it])
        break_method = "spectrum"
    else:
        nu_m, nu_c = compute_break_frequencies_from_spectrum(
            jet, P, break_times, n_freq_bins=n_freq_bins_breaks
        )
        break_method = "spectrum"

    return {
        "library": library,
        "median_params": copy.deepcopy(median_params),
        "lc_times": lc_times,
        "light_curves": light_curves,
        "observations": df,
        "checkpoints": checkpoints,
        "spectrum": {
            "times": epochs,
            "nu": spectrum_nu,
            "fnu": spectrum_fnu,
            "observations": epoch_observations,
        },
        "breaks": {
            "times": break_times,
            "nu_m": np.asarray(nu_m, dtype=float),
            "nu_c": np.asarray(nu_c, dtype=float),
            "method": break_method,
        },
    }


def save_plot_data(path, plot_data):
    """Write plot arrays so a later plotting pass does not evaluate the model."""
    with open(path, "wb") as handle:
        pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_plot_data(path):
    """Load arrays written by :func:`save_plot_data`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)
