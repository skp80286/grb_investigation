"""
Multinest script for Jetsimpy EP model with CMLP priors.
Sample command: mpirun -n 15 --oversubscribe python multinest_jetsimpy_EP_cmlp.py \
    --jetType tophat -z 0.36 --livepoints 200 --fullobsfile data/GRB230812B_modeling_v2.csv \
        --plot-spectrum --plot-break-frequencies --plot-spectrum --priors 230812B \
            --label v2theirs3 --obsfile data/GRB230812B_modeling_v2_theirs.csv
"""

import json
import os
import random
import datetime
import time
import warnings

import sys
import re

import jetsimpy
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymultinest
import pymultinest.analyse as analyse

import astropy
from astropy.cosmology import Planck15 as cosmo
from scipy import stats
from scipy.optimize import curve_fit, minimize, newton
import logging
import argparse
import requests
import sys
import warnings

from jetsimpy_plot import (
    SPECTRUM_PLOT_TIME_EPOCHS,
    break_frequency_evolution_plot,
    build_spectrum_epoch_observations,
    lc_plot,
    model,
    residual_plot,
    spectrum_plot,
)
from print_params import format_parameters_table, format_dict_table
from priors import priors_map, priors_generic
from lc_plot_settings import lc_plot_settings_default, lc_plot_settings_map

from mpi4py import MPI


#### telegram:
tele_token = os.environ["Tele_GITbot_Token"]
chat_id = os.environ["Tele_Transient_chat_id"]


def params_for_jetsimpy_model(params):
    """Copy params for model(); with --use-ksi, set logthv from logksi + logthc (thv = xi * thc)."""
    p = dict(params)
    if args.use_ksi and "logksi" in p:
        p["logthv"] = p["logksi"] + p["logthc"]
        del p["logksi"]
    return p


def Tele_alert(tele_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{tele_token}/sendMessage?chat_id={chat_id}&text={message}"
        print(requests.get(url).json())
    except Exception as e:
        print("Error in sending message : " + str(e))


######################


def log_prior(cube, ndim, nparams):
    for i, name in enumerate(param_names):
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            continue
        if name == "thc" or name == "thv":
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = np.arccos(np.cos(pmin) - cube[i] * (np.cos(pmin) - np.cos(pmax)))
        else:
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = pmin + (pmax - pmin) * cube[i]  # scale [0,1] to [min,max]


################################################

maxllh = -1e6
count = 0


def build_obs_weights(filt_values, band_targets=None):
    """
    Build per-observation weights so each band's cumulative weight matches target.
    If band_targets is None, each identified band gets cumulative weight 1.
    """
    filt_arr = np.asarray(filt_values, dtype=str)
    n_obs = len(filt_arr)
    if n_obs == 0:
        return np.array([], dtype=float)

    band_labels = np.array(filt_arr, copy=True)
    if band_targets is not None:
        # Use the configured band key (prefix match) for grouping when provided.
        sorted_keys = sorted(band_targets.keys(), key=len, reverse=True)
        for i, band in enumerate(filt_arr):
            for key in sorted_keys:
                if band.startswith(key):
                    band_labels[i] = key
                    break

    obs_weights_local = np.zeros(n_obs, dtype=float)
    unique_bands, counts = np.unique(band_labels, return_counts=True)
    for band_name, count in zip(unique_bands, counts):
        cumulative_weight = (
            float(band_targets.get(band_name, 1.0)) if band_targets is not None else 1.0
        )
        per_obs_weight = cumulative_weight / float(count)
        obs_weights_local[band_labels == band_name] = per_obs_weight
    return obs_weights_local


def log_likelihood(cube, ndim, nparams):
    # in log space
    params = {name: cube[i] for i, name in enumerate(param_names)}
    for name in priors_uniform.keys():
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            params[name] = priors_uniform[name]["low"]
    params["jetType"] = args.jetType
    params["z"] = args.redshift

    try:
        model_flux = model(obs_time, obs_nu, params_for_jetsimpy_model(params))
    except Exception as e:
        logger.debug("log_likelihood: model failed (%s); returning penalty llh", e)
        return -1e100

    # Residuals and chi2 term
    residual = 1 - (model_flux / obs_flux)
    if np.any(obs_ul):
        # For upper limits: very high residual if model exceeds limit, else zero
        residual[obs_ul] = np.where(
            model_flux[obs_ul] > obs_flux[obs_ul],
            1e10,  # Infinite penalty if model exceeds the upper limit
            0.0,  # Zero penalty if model respects the limit
        )

    try:
        chi2_terms = 0.5 * (residual / err) ** 2
    except RuntimeWarning as e:
        print("Residual min/max:", np.nanmin(residual), np.nanmax(residual))
        print("Err min/max:", np.nanmin(err), np.nanmax(err))
        print("Ratio min/max:", np.nanmin(ratio), np.nanmax(ratio))
        print("Any inf residual:", np.isinf(residual).any())
        print("Any inf err:", np.isinf(err).any())

        warnings.filterwarnings("error", category=RuntimeWarning)
        print(f"Fatal RuntimeWarning: {e}")
        sys.exit(1)

    if args.use_band_weights or args.equal_band_weights:
        chi2_terms *= obs_weights
    chi2 = np.sum(chi2_terms)
    # logger.info(f"weighted residuals={residuals}")
    # llh = -0.5 * np.sum(residuals**2)
    llh = logdet - chi2
    global count
    count += 1
    if count % 1000 == 0:
        logger.info(f"count={count}, llh={llh}")
    global maxllh
    if llh > maxllh:
        maxllh = llh
        # params_str = ", ".join( f"{param}={cube[i]:.8f}" for i, param in enumerate(param_names))
        # logger.info(f"Log-likelihood: {llh}, {params_str}, \nobs_flux={obs_flux}\n, model_flux={model_flux}")
    return llh


def numbers_only_title(val, q, labels):
    return f"{val:.3f}$^{{+{q[1] - val:.2f}}}_{{-{val - q[0]:.2f}}}$"


################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


parser = argparse.ArgumentParser()
parser.add_argument("--jetType", type=str, default="tophat")
parser.add_argument("--livepoints", type=int, default=500)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--obsfile", type=str, default="multinest_EP/mcmc_df_trunc.csv")
parser.add_argument("--fullobsfile", type=str, default="multinest_EP/mcmc_df.csv")
parser.add_argument("--alert", action="store_true", help="Enable telegram alert.")
parser.add_argument(
    "--post_process_only",
    action="store_true",
    help="Only postprocess using previous analysis.",
)
parser.add_argument(
    "--use_band_weights",
    action="store_true",
    help="weights for residuals in diff bands.",
)
parser.add_argument(
    "--equal_band_weights",
    action="store_true",
    help="set cumulative weight to 1 for each observed band.",
)
parser.add_argument(
    "--use-ksi",
    action="store_true",
    help=(
        "Sample log10(xi) with xi = theta_v/theta_c instead of log10(theta_v); "
        "theta_v is set to xi * theta_c."
    ),
)
parser.add_argument("-z", "--redshift", type=float, default=0, help="redshift")
parser.add_argument(
    "--plot-residuals",
    type=str,
    default="",
    help=(
        "Comma-separated filter names (Filt column) for residual plots after the "
        'light-curve plot, e.g. "g,r,i" or "X-ray(10keV)". Ignored if empty.'
    ),
)
parser.add_argument(
    "--plot-spectrum",
    action="store_true",
    help=(
        "After the light-curve plot, plot modeled spectrum vs frequency at "
        "log-spaced times (1 s to 1e6 s) using jet.Flux() and median parameters."
    ),
)
parser.add_argument(
    "--plot-break-frequencies",
    action="store_true",
    help=(
        "After the light-curve plot, plot evolution of synchrotron break frequencies "
        "nu_m and nu_c using median inferred parameters (analytical scalings if jet "
        "type is tophat, otherwise inferred from Jet.Flux synchrotron spectra)."
    ),
)

parser.add_argument(
    "--priors",
    type=str,
    default="generic",
    help="Priors set to use (e.g. 'generic', 'dirty_fireball' or 'structured_offaxis')",
)
parser.add_argument(
    "--lc-plot-settings",
    type=str,
    default="default",
    choices=sorted(lc_plot_settings_map),
    help="Named light-curve plot settings from lc_plot_settings.py (default: default).",
)


parser.add_argument(
    "--use_ul",
    action="store_true",
    help="Consider upper-limit (UL) rows in the likelihood (default: False)",
)
parser.add_argument(
    "--hide-z-text",
    action="store_true",
    help="Omit the textbox listing fitted parameter values on the light-curve plot.",
)
args = parser.parse_args()

np.random.seed(12)

# read obs csv
file = args.obsfile

# Set up the output directory and logging
obsfile_name = os.path.basename(args.obsfile)
target_name = obsfile_name.split("_", 1)[0]
output_root = "output"
base_prefix = f"multinest_{target_name}_{args.jetType}"

os.makedirs(output_root, exist_ok=True)
if args.label:
    basedir = os.path.join(output_root, f"{base_prefix}_{args.label}")
else:
    existing_indices = []
    for dirname in os.listdir(output_root):
        match = re.fullmatch(rf"{re.escape(base_prefix)}_(\d+)", dirname)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    basedir = os.path.join(output_root, f"{base_prefix}_{next_index:03d}")
os.makedirs(basedir, exist_ok=True)
outputfiles_basename = basedir + f"/multinest_"
# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{outputfiles_basename}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info(f"Commandline: {' '.join(sys.argv)}")

# setup for the params
"""
# Tophat
priors_uniform = {
    "loge0": {"low": 53, "high": 54},
    "logepsb": {"low": -3, "high": -1.8},
    "logepse": {"low": -1.3, "high": -0.8},
    "logn0": {"low": -1.8, "high": 2.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 2.2},
    "s": {"low": 2, "high": 2},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0.0, "high": 0.0}, # 
}

# Powerlaw
priors_uniform = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1, "high": 8},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0.0, "high": 0.0}, # 
}

2026-04-25 06:12:13,509 - INFO - logksi: {'low': -2.0, 'high': 1.0}
2026-04-25 06:12:13,509 - INFO - logepsb: {'low': -8, 'high': -1}
2026-04-25 06:12:13,509 - INFO - p: {'low': 2.01, 'high': 3.0}
2026-04-25 06:12:13,509 - INFO - logepse: {'low': -2, 'high': -0.5}
2026-04-25 06:12:13,509 - INFO - s: {'low': 0, 'high': 0}
2026-04-25 06:12:13,509 - INFO - logn0: {'low': -2.0, 'high': 0.0}
2026-04-25 06:12:13,509 - INFO - loglf: {'low': 1, 'high': 10}
2026-04-25 06:12:13,509 - INFO - logthc: {'low': -3.0, 'high': -0.5}
2026-04-25 06:12:13,509 - INFO - logA: {'low': 0.0, 'high': 0.0}

priors_uniform_dirty_fb = {
    "loge0": {"low": 48, "high": 55},
    "logepsb": {"low": -2, "high": -2},
    "logepse": {"low": -1.5, "high": -0.5},
    "logn0": {"low": -6.0, "high": 0.0},
    "logthc": {"low": -1.0, "high": 0},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians; omitted when --use-ksi
    "logksi": {
        "low": 0.0,
        "high": 1,
    },  # log10(theta_v/theta_c); only sampled with --use-ksi
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1, "high": 8},
    "loglf": {"low": 6, "high": 6},
    "logA": {"low": 0.0, "high": 0.0},  #
}

"""

# Select priors based on command-line argument. Default to generic if unknown.
lc_plot_settings = lc_plot_settings_map.get(
    args.lc_plot_settings, lc_plot_settings_default
)
priors_uniform = priors_map.get(args.priors, priors_generic)

if args.use_ksi:
    priors_uniform.pop("logthv", None)
else:
    priors_uniform.pop("logksi", None)

if args.jetType != "powerlaw":
    priors_uniform["s"]["low"] = 0
    priors_uniform["s"]["high"] = 0

param_names = [
    key
    for key in priors_uniform.keys()
    if priors_uniform[key]["low"] != priors_uniform[key]["high"]
]
# param_names = ["loge0", "logepsb", "s", "logn0", "thc", "thv", "p"]
n_params = len(param_names)
param_names_math = {
    "loge0": r"$\log_{10}(E_{K,iso})$",
    "logepsb": r"$\log_{10}(\epsilon_{B})$",
    "logepse": r"$\log_{10}(\epsilon_{e})$",
    "logn0": r"$\log_{10}(n_0)$",
    "logthc": r"$\log_{10}(\theta_{c})$",
    "logthv": r"$\log_{10}(\theta_{v})$",
    "logksi": r"$\log_{10}(\xi)$",
    "p": r"$p$",
    "s": r"$s$",
    "loglf": r"$\log_{10}(\Gamma_0)$",
    "logA": r"$\log_{10}(A)$",
    "k": r"$k$",
}
param_names_display = [param_names_math[p] for p in param_names]
# param_names_greeks = [r"$\log_{10}(E_{K,iso})", r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\log_{10}(n0)$", r"$\theta_{c}$", r"$\theta_{v}$", r"$\Gamma_0$"]
## TODO
param_names_greeks = [
    r"$\log_{10}(E_{K,iso})$",
    r"$\log_{10}(\epsilon_{B})$",
    r"$\log_{10}(\epsilon_{e})$",
    r"$\theta_{c}$",
    r"$\theta_{v}$",
]
# param_names_greeks = [r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\theta_{c}$", r"$\theta_{v}$", r"$\Gamma_0$"]

band_weights = {"X-ray": 3, "radio": 3, "i": 2}
if not args.post_process_only:
    logger.info(f"######### Priors: #########")
    for key, value in priors_uniform.items():
        logger.info(f"{key}: {value}")
    logger.info(f"###########################")

    data = pd.read_csv(file)
    required_columns = ["Times", "Freqs", "Fluxes", "FluxErrs"]

    if not all(col in data.columns for col in required_columns):
        logger.error(f"Input CSV file is missing required columns: {required_columns}")
        raise ValueError(
            f"Input CSV file is missing required columns: {required_columns}"
        )
    data = data[data["Fluxes"] > 0]  # taking detections only
    if len(data) == 0:
        logger.error("len of the input data is zero")

    obs_time = data["Times"].to_numpy()  # time in seconds
    obs_nu = data["Freqs"].to_numpy()  # in Hz
    obs_flux = data["Fluxes"].to_numpy()  # in mJy
    obs_flux_err = data["FluxErrs"].to_numpy()  # in mJy

    if "UL" in data.columns:
        obs_ul = (
            data["UL"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin({"Y", "YES", "TRUE", "T", "1"})
            .to_numpy()
        )
    else:
        obs_ul = np.zeros(len(obs_flux), dtype=bool)

    # Only consider upper limits in likelihood if requested via CLI
    if not args.use_ul:
        obs_ul = np.zeros(len(obs_flux), dtype=bool)
        logger.info(
            "Upper limits present in file but will be ignored (use --use_ul to enable)."
        )
    else:
        logger.info(
            "Considering upper limits in likelihood as requested (--use_ul set)."
        )

    # Log-determinant term
    err = np.where(obs_flux > 0, obs_flux_err / obs_flux, np.inf)
    err = np.where(err <= 0, np.finfo(float).tiny, err)
    logdet = np.sum(np.log(2.0 * np.pi * err**2))

    obs_weights = np.ones(len(obs_flux), dtype=float)
    if args.use_band_weights and args.equal_band_weights:
        logger.warning(
            "Both --use_band_weights and --equal_band_weights are set; "
            "using band_weights cumulative targets."
        )
    if args.use_band_weights:
        obs_weights = build_obs_weights(data["Filt"], band_targets=band_weights)
        logger.info(
            f"Applied configured cumulative band weights: {band_weights}; "
            f"total weight={np.sum(obs_weights):.3f}"
        )
    elif args.equal_band_weights:
        obs_weights = build_obs_weights(data["Filt"], band_targets=None)
        logger.info(
            "Applied equal cumulative band weights (1 per band); "
            f"total weight={np.sum(obs_weights):.3f}"
        )

    maxllh = -1e6

    obs_table = data[["Times", "Filt", "Fluxes"]].rename(
        columns={"Times": "time_s", "Filt": "band", "Fluxes": "flux_mJy"}
    )
    logger.info(
        "Observations table (%d total):\n%s",
        len(obs_table),
        obs_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
    )
    logger.info(f"Observations file has {len(obs_flux)} records.")
    logger.info(f"Starting MultiNest run with {n_params} parameters: {param_names}.")
    result = None
    try:
        result = pymultinest.run(
            log_likelihood,
            log_prior,
            n_params,
            outputfiles_basename=outputfiles_basename,
            n_live_points=args.livepoints,
            sampling_efficiency=0.8,
            evidence_tolerance=0.3,
            importance_nested_sampling=True,
            resume=False,
            verbose=True,
        )
        logger.info(f"Finished MultiNest run. Process {rank}.")
    except Exception as e:
        logger.error(f"Caught exception in multicast run! {e}")
        raise

if rank != 0:
    sys.exit(0)
if rank == 0:  # Only one process does the analysis
    logger.info(f"Process {rank}: Analyzing results.")
    time.sleep(3)  # artificial barrier
    # comm.Barrier() # All processes will wait here until every process reaches this point
    print(f"Process {rank}: Finished pre-barrier work, entering barrier.")

    # Create an Analyzer object
    analyzer = analyse.Analyzer(n_params, outputfiles_basename=outputfiles_basename)

    # Get the Bayesian evidence
    # logZ = analyzer.get_log_evidence()        # ln(Z)
    # logZ_err = analyzer.get_log_evidence_err()
    stats = analyzer.get_stats()
    logger.info(f"stats={stats}")
    lnZ = stats["nested importance sampling global log-evidence"]
    lnZErr = stats["nested importance sampling global log-evidence error"]
    # calculate Bayesian Information Criterion (BIC) used for comparing models
    # Determine number of observations: use obs_flux if available, otherwise count CSV lines

    try:
        num_obs = len(pd.read_csv(args.obsfile)) - 1
    except Exception:
        num_obs = 1
        logger.warning(
            f"obs_flux was None or empty; using line count from CSV: num_obs={num_obs}"
        )

    bic = astropy.stats.bayesian_info_criterion(lnZ, n_params, num_obs)
    logger.info(
        f"lnZ={lnZ:.4f} lnZErr={lnZErr:.4f}, n_params={n_params:.4f}, num_obs={num_obs}, BIC={bic:.4f}"
    )

    # Get the best-fit parameters (highest likelihood point)
    bestfit_params = analyzer.get_best_fit()

    # Print the best-fit parameters
    params_str = ", ".join(
        f"{param}={bestfit_params['parameters'][i]:.8f}"
        for i, param in enumerate(param_names)
    )
    logger.info(f"Best-fit parameters: {params_str}")
    json.dump(
        bestfit_params["parameters"], open(outputfiles_basename + "params.json", "w")
    )

    # corner plot
    logger.info("Making corner plot")
    flat_samples = analyzer.get_equal_weighted_posterior()[:, :-1]
    medians = np.median(flat_samples, axis=0)

    covariance = np.cov(flat_samples, rowvar=False)
    sigma = np.sqrt(np.diagonal(covariance))
    lower_bounds = medians - 3 * sigma
    upper_bounds = medians + 3 * sigma
    relative_sigma = np.abs(sigma / medians)
    sig3_flat_samples = flat_samples[
        np.all((flat_samples >= lower_bounds) & (flat_samples <= upper_bounds), axis=1)
    ]

    logger.info(
        f"Total flat samples from eq weighted posterior: {len(flat_samples)}, total sigma3 smaples: {len(sig3_flat_samples)}"
    )
    # corner
    fig = corner.corner(
        flat_samples,
        labels=param_names_display,
        show_titles=True,
        title_fmt=".2f",
        truths=medians,
        title_kwargs={"fontsize": 24},
        # add smooth
        smooth=2,
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 24},
        labelpad=0.1,
        hist_kwargs={"density": True, "alpha": 0.5},
    )
    # Hack to strip off the label from axis title
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            t = ax.title
            fontsize = t.get_fontsize()
            title = re.sub(r".*?=\s*", "", title)
            ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=16)

    # save the figure
    corner_plot_file = basedir + "/multinest_corner.pdf"
    logger.info(f"Saving corner plot: {corner_plot_file}")
    plt.savefig(
        corner_plot_file,
        dpi=300,
        format="pdf",
        bbox_inches="tight",
    )
    corner_plot_file = basedir + "/multinest_corner.png"
    logger.info(f"Saving corner plot: {corner_plot_file}")
    plt.savefig(
        corner_plot_file,
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # light curve fitting plot
    logger.info(f"Creating lightcurve plot")
    median_params = {}
    median_params["jetType"] = args.jetType
    median_params["z"] = args.redshift
    # params['logepse']=-1
    # params['loglf']=np.log10(200.0)
    for key, value in priors_uniform.items():
        if value["low"] == value["high"]:
            median_params[key] = value["low"]
    logger.info(format_dict_table(median_params, "Fixed parameters"))
    for i, value in enumerate(medians):
        median_params[param_names[i]] = value
    rel_sigma_params = {}
    for i, value in enumerate(relative_sigma):
        rel_sigma_params[param_names[i]] = value
    logger.info(
        f"Inferred parameters: \n{format_parameters_table(median_params, rel_sigma_params, priors_uniform)}"
    )

    sig3_params = []
    # for i in np.random.randint(len(sig3_flat_samples), size=50):
    # for sample in len(flat_samples):
    for i in np.random.randint(len(flat_samples), size=100):
        # sample = sig3_flat_samples[i]
        sample = flat_samples[i]
        params = {}
        params["jetType"] = args.jetType
        params["z"] = args.redshift
        # params['logepse']=-1
        # params['loglf']=np.log10(200.0)
        for j, value in enumerate(sample):
            params[param_names[j]] = value
        for key, value in priors_uniform.items():
            if value["low"] == value["high"]:
                params[key] = value["low"]
        sig3_params.append(params_for_jetsimpy_model(params))
    logger.info(f"3 Sigma parameters: {sig3_params[:10]}")

    lc_plot(
        basedir,
        params_for_jetsimpy_model(median_params),
        sig3_params,
        observed_data=args.fullobsfile,
        hide_z_text=args.hide_z_text,
        plot_settings=lc_plot_settings,
    )

    if args.plot_spectrum:
        df_spectrum_obs = pd.read_csv(args.fullobsfile)
        epoch_obs = build_spectrum_epoch_observations(
            df_spectrum_obs, SPECTRUM_PLOT_TIME_EPOCHS, dt_sec=500.0
        )
        spectrum_plot(
            basedir,
            params_for_jetsimpy_model(median_params),
            time_epochs=SPECTRUM_PLOT_TIME_EPOCHS,
            epoch_observations=epoch_obs,
        )

    if args.plot_break_frequencies:
        break_frequency_evolution_plot(
            basedir, params_for_jetsimpy_model(median_params)
        )

    if args.plot_residuals.strip():
        median_for_plot = params_for_jetsimpy_model(median_params)
        for filt in (s.strip() for s in args.plot_residuals.split(",") if s.strip()):
            logger.info("Creating residual plot for filter %r", filt)
            residual_plot(
                basedir,
                median_for_plot,
                observed_data=args.fullobsfile,
                filt=filt,
            )

    if args.alert:
        message = (
            f"Run is complete. maxllh={maxllh:.2f}. Please check the parameters:"
            + "\n"
            + params_str
        )
        Tele_alert(tele_token, chat_id, message)
