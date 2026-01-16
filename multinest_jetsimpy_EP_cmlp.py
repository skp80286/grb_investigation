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

from jetsimpy_plot import model, lc_plot
from print_params import format_parameters_table, format_dict_table

from mpi4py import MPI


#### telegram:
tele_token = os.environ["Tele_GITbot_Token"]
chat_id = os.environ["Tele_Transient_chat_id"]

def Tele_alert(tele_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{tele_token}/sendMessage?chat_id={chat_id}&text={message}"
        print(requests.get(url).json())
    except Exception as e:
        print("Error in sending message : " + str(e))

######################

def log_prior(cube, ndim, nparams):
    for i, name in enumerate(param_names):
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]: continue
        if name == 'thc' or name == 'thv':
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = np.arccos(np.cos(pmin) - cube[i] * (np.cos(pmin) - np.cos(pmax)))
        else:
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = pmin + (pmax - pmin) * cube[i]  # scale [0,1] to [min,max]

################################################

def log_likelihood_vishwa(cube, ndim, nparams):
    params = {name: cube[i] for i, name in enumerate(param_names)}
    for name in priors_uniform.keys():
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            params[name] = priors_uniform[name]["low"]
    params['jetType'] = args.jetType
    params['z'] = args.redshift

    model_flux = model(obs_time, obs_nu, params)
    #log_obs_flux_err = obs_flux_err / obs_flux

    residuals = np.log(obs_flux / model_flux) / err # log_obs_flux_err
    llf = -0.5 * np.sum(residuals**2)
    return llf


maxllh = -1e6
def log_likelihood(cube, ndim, nparams):
    # in log space
    params = {name: cube[i] for i, name in enumerate(param_names)}
    for name in priors_uniform.keys():
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            params[name] = priors_uniform[name]["low"]
    params['jetType'] = args.jetType
    params['z'] = args.redshift

    model_flux = model(obs_time, obs_nu, params)

    # Residuals and chi2 term
    residual = 1 - (model_flux / obs_flux)

    chi2 = 0.5*np.sum((residual/err) ** 2)

    #logger.info(f"original residuals={residuals}")
    if args.use_band_weights:
        chi2 *= obs_weights
    #logger.info(f"weighted residuals={residuals}")
    #llh = -0.5 * np.sum(residuals**2)
    llh = logdet - chi2 
    global maxllh
    if llh > maxllh:
        maxllh = llh
        #params_str = ", ".join( f"{param}={cube[i]:.8f}" for i, param in enumerate(param_names))
        #logger.info(f"Log-likelihood: {llh}, {params_str}, \nobs_flux={obs_flux}\n, model_flux={model_flux}")
    return llh

def log_likelihood_2(cube, ndim, nparams):
    params = {name: cube[i] for i, name in enumerate(param_names)}
    for name in priors_uniform.keys():
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            params[name] = priors_uniform[name]["low"]
    params['jetType'] = args.jetType
    params['z'] = args.redshift

    #if params["thv"]/params["thc"] > 1.5:
        #return -np.inf
    model_flux = model(obs_time, obs_nu, params)

    log_obs_flux_err = obs_flux_err / obs_flux
    residuals = np.log(obs_flux / model_flux) / log_obs_flux_err
    llf = -0.5 * np.sum(residuals**2)
    return llf

def numbers_only_title(val, q, labels):
    return f"{val:.3f}$^{{+{q[1]-val:.2f}}}_{{-{val-q[0]:.2f}}}$"

################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


parser = argparse.ArgumentParser()
parser.add_argument('--jetType', type=str, default='tophat')
parser.add_argument('--livepoints', type=int, default=500)
parser.add_argument('--label', type=str, default='')
parser.add_argument('--obsfile', type=str, default='multinest_EP/mcmc_df_trunc.csv')
parser.add_argument('--fullobsfile', type=str, default='multinest_EP/mcmc_df.csv')
parser.add_argument('--alert', action='store_true', help='Enable telegram alert.')
parser.add_argument('--post_process_only', action='store_true', help='Only postprocess using previous analysis.')
parser.add_argument('--use_band_weights', action='store_true', help='weights for residuals in diff bands.')
parser.add_argument('-z', '--redshift', type=float, default=0, help='redshift')

args = parser.parse_args()

np.random.seed(12)

# read obs csv
file = args.obsfile

# Set up the output directory and logging
basedir =  f"output/multinest_{args.label}"
os.makedirs(basedir, exist_ok=True)
outputfiles_basename = (
    basedir + f"/jetsimpy_"
)
# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{outputfiles_basename}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Commandline: {' '.join(sys.argv)}")

# setup for the params
"""
Tophat:
priors_uniform = {
    "loge0": {"low": 53, "high": 54},
    "logepsb": {"low": -3, "high": -1.8},
    "logepse": {"low": -1.3, "high": -0.8},
    "logn0": {"low": -1.8, "high": 2.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 2.2},
    "s": {"low": 1, "high": 8},
    "loglf": {"low": 3.0, "high": 3.0},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}

#Powerlaw:
priors_uniform = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 3.00},
    "s": {"low": 1, "high": 8},
    "loglf": {"low": 3.0, "high": 3.0},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}


#gaussian
priors_uniform = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -2},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 2.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 2.2},
    "s": {"low": 2, "high": 2},
    "loglf": {"low": 3.0, "high": 3.0},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}
priors_uniform = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.07, "high": 2.07},
    "s": {"low": 2, "high": 2},
    "loglf": {"low": 3.0, "high": 3.0},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}
"""
priors_uniform = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1.0},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -5, "high": -0.5},  # radians
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 2, "high": 2},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0.0, "high": 0.0}, # 
}

if args.jetType == 'tophat': 
    priors_uniform["s"]["low"]=0
    priors_uniform["s"]["high"]=0
param_names = [key for key in priors_uniform.keys() if priors_uniform[key]["low"] != priors_uniform[key]["high"]]
#param_names = ["loge0", "logepsb", "s", "logn0", "thc", "thv", "p"]
n_params = len(param_names)
param_names_math = {"loge0":r"$\log_{10}(E_{K,iso})$", "logepsb":r"$\log_{10}(\epsilon_{B})$",  "logepse": r"$\log_{10}(\epsilon_{e})$", "logn0": r"$\log_{10}(n_0)$", "logthc": r"$\log_{10}(\theta_{c})$", "logthv": r"$\log_{10}(\theta_{v})$", "p":r"$p$", "s": r"$s$", "loglf": r"$\log_{10}(\Gamma_0)$", "logA": r"$\log_{10}(A)$", "k": r"$k$"}
param_names_display = [param_names_math[p] for p in param_names]
#param_names_greeks = [r"$\log_{10}(E_{K,iso})", r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\log_{10}(n0)$", r"$\theta_{c}$", r"$\theta_{v}$", r"$\Gamma_0$"]
## TODO
param_names_greeks = [r"$\log_{10}(E_{K,iso})$",r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\theta_{c}$", r"$\theta_{v}$"]
#param_names_greeks = [r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\theta_{c}$", r"$\theta_{v}$", r"$\Gamma_0$"]

band_weights={'X-ray':3, 'radio': 3, 'i': 2}
if not args.post_process_only:
    logger.info(f"######### Priors: #########")
    for key, value in priors_uniform.items():
        logger.info(f"{key}: {value}")
    logger.info(f"###########################")

    data = pd.read_csv(file)
    required_columns = ["Times", "Freqs", "Fluxes", "FluxErrs"]
    
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Input CSV file is missing required columns: {required_columns}")
        raise ValueError(f"Input CSV file is missing required columns: {required_columns}")
    data = data[data["Fluxes"] > 0]  # taking detections only
    if len(data) == 0:
        logger.error("len of the input data is zero")
    obs_time = data["Times"].to_numpy()  # time in seconds
    obs_nu = data["Freqs"].to_numpy()  # in Hz
    obs_flux = data["Fluxes"].to_numpy()  # in mJy
    obs_flux_err = data["FluxErrs"].to_numpy()  # in mJy
    # Log-determinant term
    err = obs_flux_err / obs_flux

    logdet = np.sum(np.log(2.0 * np.pi * err**2))

    obs_weights = np.ones(len(obs_flux))
    if args.use_band_weights:
        for i,band in enumerate(data["Filt"]):
            for key, value in band_weights.items():
                if band.startswith(key):
                    obs_weights[i] = value
                    logger.info(f"band={band}, weight={value}")

    maxllh = -1e6

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
        logger.error(f'Caught exception in multicast run! {e}')
        raise

if rank != 0: sys.exit(0)
if rank == 0: # Only one process does the analysis
    logger.info(f"Process {rank}: Analyzing results.")
    time.sleep(3) # artificial barrier
    #comm.Barrier() # All processes will wait here until every process reaches this point
    print(f"Process {rank}: Finished pre-barrier work, entering barrier.")


    # Create an Analyzer object
    analyzer = analyse.Analyzer(n_params, outputfiles_basename=outputfiles_basename)

    # Get the Bayesian evidence
    #logZ = analyzer.get_log_evidence()        # ln(Z)
    #logZ_err = analyzer.get_log_evidence_err()
    stats = analyzer.get_stats()
    logger.info(f"stats={stats}")
    lnZ = stats['nested importance sampling global log-evidence']
    lnZErr = stats['nested importance sampling global log-evidence error']
    # calculate Bayesian Information Criterion (BIC) used for comparing models
    bic = astropy.stats.bayesian_info_criterion(lnZ, n_params, 28)#len(obs_flux))
    logger.info(f"lnZ={lnZ:.4f} lnZErr={lnZErr:.4f}, n_params={n_params:.4f}, num_obs=28, BIC={bic:.4f}")

    # Get the best-fit parameters (highest likelihood point)
    bestfit_params = analyzer.get_best_fit()

    # Print the best-fit parameters
    params_str = ", ".join(
        f"{param}={bestfit_params['parameters'][i]:.8f}"
        for i, param in enumerate(param_names)
    )
    logger.info(f"Best-fit parameters: {params_str}")
    json.dump(bestfit_params["parameters"], open(outputfiles_basename + "params.json", "w"))

    # corner plot
    logger.info("Making corner plot")
    flat_samples = analyzer.get_equal_weighted_posterior()[:, :-1]
    medians = np.median(flat_samples, axis=0)

    covariance = np.cov(flat_samples, rowvar=False)
    sigma = np.sqrt(np.diagonal(covariance))
    lower_bounds = medians - 3 * sigma
    upper_bounds = medians + 3 * sigma
    relative_sigma = np.abs(sigma/medians)
    sig3_flat_samples = flat_samples[
        np.all((flat_samples >= lower_bounds) & (flat_samples <= upper_bounds), axis=1)
    ]

    logger.info(f'Total flat samples from eq weighted posterior: {len(flat_samples)}, total sigma3 smaples: {len(sig3_flat_samples)}')
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
    median_params['jetType']=args.jetType
    median_params['z']=args.redshift
    #params['logepse']=-1
    #params['loglf']=np.log10(200.0)
    for key, value in priors_uniform.items():
        if value["low"] == value["high"]:
            median_params[key] = value["low"]
    logger.info(format_dict_table(median_params, "Fixed parameters"))
    for i, value in enumerate(medians):
        median_params[param_names[i]] = value
    rel_sigma_params = {}
    for i, value in enumerate(relative_sigma):
        rel_sigma_params[param_names[i]] = value
    logger.info(f'Inferred parameters: \n{format_parameters_table(median_params, rel_sigma_params, priors_uniform)}')

    sig3_params = []
    #for i in np.random.randint(len(sig3_flat_samples), size=50):
    #for sample in len(flat_samples):
    for i in np.random.randint(len(flat_samples), size=100):
        #sample = sig3_flat_samples[i]
        sample = flat_samples[i]
        params = {}
        params['jetType']=args.jetType
        params['z']=args.redshift
        #params['logepse']=-1
        #params['loglf']=np.log10(200.0)
        for j, value in enumerate(sample):
            params[param_names[j]] = value
        for key, value in priors_uniform.items():
            if value["low"] == value["high"]:
                params[key] = value["low"]
        sig3_params.append(params)
    logger.info(f'3 Sigma parameters: {sig3_params[:10]}')

    lc_plot(basedir, median_params, sig3_params, observed_data=args.fullobsfile)

    if args.alert:
        message = f"Run is complete. maxllh={maxllh:.2f}. Please check the parameters:" + "\n" + params_str
        Tele_alert(tele_token, chat_id, message)

