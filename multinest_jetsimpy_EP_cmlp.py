import json
import os
import random
import datetime
import warnings

import sys

import jetsimpy
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymultinest
import pymultinest.analyse as analyse

# import redback
from astropy.cosmology import Planck15 as cosmo
from scipy import stats
from scipy.optimize import curve_fit, minimize, newton
import logging
import argparse
import requests

from jetsimpy_plot import model, lc_plot

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

maxllh = -1e6
def log_likelihood(cube, ndim, nparams):
    # in log space
    params = {name: cube[i] for i, name in enumerate(param_names)}
    for name in priors_uniform.keys():
        if priors_uniform[name]["low"] == priors_uniform[name]["high"]:
            params[name] = priors_uniform[name]["low"]
    params['jetType'] = args.jetType
    params['z'] = args.redshift

    #if params["thv"]/params["thc"] > 1.5:
        #return -np.inf
    model_flux = model(obs_time, obs_nu, params)
    #print(f'obs_time={obs_time},\nobs_nu={obs_nu},\nobs_flux={obs_flux},\nmodel_flux={model_flux}')
    # log_model_flux = np.log(model_flux)
    # log_obs_flux = np.log(obs_flux)
    # log_obs_flux_err = obs_flux_err / obs_flux
    residuals = np.log(obs_flux / model_flux) #/ log_obs_flux_err
    #logger.info(f"original residuals={residuals}")
    if args.use_band_weights:
        residuals *= obs_weights
    #logger.info(f"weighted residuals={residuals}")
    llh = -0.5 * np.sum(residuals**2)
    global maxllh
    if llh > maxllh:
        maxllh = llh
        params_str = ", ".join( f"{param}={cube[i]:.8f}" for i, param in enumerate(param_names)) 
        logger.info(f"Log-likelihood: {llh}, {params_str}, \nobs_flux={obs_flux}\n, model_flux={model_flux}")
    return llh

################################################

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
priors_uniform = {
    "loge0": {"low": 47.0, "high": 53.0},
    "logepsb": {"low": -6.0, "high": -1.0},
    "logepse": {"low": -3.0, "high": -0.5},
    "logn0": {"low": -4.0, "high": 0.0},
    "thc": {"low": 0.01, "high": 0.5},  # radians
    "thv": {"low": 0.01, "high": 0.8},  # radians
    "p": {"low": 2.01, "high": 2.9},
    "s": {"low": 5, "high": 8},
    "lf": {"low": 50, "high": 500},
    "A": {"low": 0.0, "high": 1.0},
}
2025-08-15 06:52:07,431 - INFO - loge0: {'low': 51, 'high': 55}
2025-08-15 06:52:07,431 - INFO - logepsb: {'low': -4.0, 'high': -1.0}
2025-08-15 06:52:07,431 - INFO - logepse: {'low': -1.5, 'high': 0}
2025-08-15 06:52:07,431 - INFO - logn0: {'low': -3, 'high': 0}
2025-08-15 06:52:07,431 - INFO - logthc: {'low': -1.25, 'high': -2.0}
2025-08-15 06:52:07,431 - INFO - logthv: {'low': -0.5, 'high': -2.0}
2025-08-15 06:52:07,431 - INFO - p: {'low': 2.0, 'high': 2.5}
2025-08-15 06:52:07,431 - INFO - s: {'low': 6.5, 'high': 6.5}
2025-08-15 06:52:07,431 - INFO - loglf: {'low': 7.1, 'high': 7.1}
2025-08-15 06:52:07,431 - INFO - A: {'low': 0.0, 'high': 0.0}

priors_uniform = {
    "loge0": {"low": 45, "high": 55},
    "logepsb": {"low": -2.0, "high": -2.0},
    "logepse": {"low": -1.0, "high": -1.0},
    "logn0": {"low": -4.0, "high": 1.0},
    "logthc": {"low": -3, "high": -0.5},  # radians
    "logthv": {"low": -3, "high": -0.5},  # radians
    "p": {"low": 2.0, "high": 2.0},
    "s": {"low": 6.0, "high": 6.0},      ## fix at 6.0
    "loglf": {"low": 10.0, "high": 10.0},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}
"""
"""
#powerlaw
priors_uniform = {
    "loge0": {"low": 51.0, "high": 56, "prior_type": "uniform"},
    "logepsb": {"low": -3.0, "high": -1.8, "prior_type": "uniform"},
    "logepse": {"low": -1.4, "high": -0.6, "prior_type": "uniform"},
    "logn0": {"low": -2.0, "high": -0.6, "prior_type": "uniform"},
    #"ln0": {"low": -4.0, "high": 1.0, "prior_type": "uniform"},
    "thc": {"low": 0.06, "high": 0.18, "prior_type": "uniform"},  # radians
    "logthv": {"low": -2, "high": -1.0},
    "p": {"low": 2.0961, "high": 2.0961, "prior_type": "uniform"},
    "s": {"low": 2, "high": 5.5, "prior_type": "uniform"},
    "loglf": {"low": 10, "high": 10},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}

"""
#tophat
priors_uniform = {
    "loge0": {"low": 51.0, "high": 55.0, "prior_type": "uniform"},
    "logepsb": {"low": -3.0, "high": -1.0, "prior_type": "uniform"},
    "logepse": {"low": -0.4, "high": -0.4, "prior_type": "uniform"},
    "logn0": {"low": -2.5, "high": -2.5, "prior_type": "uniform"},
    #"ln0": {"low": -4.0, "high": 1.0, "prior_type": "uniform"},
    "logthc": {"low": -1.6, "high": -0.8, "prior_type": "uniform"},  # radians
    "logthv": {"low": -3.2, "high": 0.0},
    "p": {"low": 2.0, "high": 2.8, "prior_type": "uniform"},
    "s": {"low": 1, "high": 6, "prior_type": "uniform"},
    "loglf": {"low": 10, "high": 10},
    "A": {"low": 0.0, "high": 0.0},  ## fix at 0
}

if args.jetType != 'powerlaw': 
    priors_uniform["s"]["low"]=0
    priors_uniform["s"]["high"]=0
param_names = [key for key in priors_uniform.keys() if priors_uniform[key]["low"] != priors_uniform[key]["high"]]
#param_names = ["loge0", "logepsb", "s", "logn0", "thc", "thv", "p"]
n_params = len(param_names)
#param_names_greeks = [r"$\log_{10}(E_{K,iso})", r"$\log_{10}(\epsilon_{B})$",  r"$\log_{10}(\epsilon_{e})$", r"$\log_{10}(n0)$", r"$\theta_{c}$", r"$\theta_{v}$", r"$p$","$s$", r"$\Gamma_0$", r"$A$"]
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
    obs_weights = np.ones(len(obs_flux))
    if args.use_band_weights:
        for i,band in enumerate(data["Filt"]):
            for key, value in band_weights.items():
                if band.startswith(key):
                    obs_weights[i] = value
                    logger.info(f"band={band}, weight={value}")

    maxllh = -1e6

    logger.info(f"Starting MultiNest run with {n_params} parameters: {param_names}.")
    try:
        pymultinest.run(
            log_likelihood,
            log_prior,
            n_params,
            outputfiles_basename=outputfiles_basename,
            n_live_points=args.livepoints,
            sampling_efficiency=0.8,
            evidence_tolerance=0.3,
            # importance_nested_sampling=False,
            resume=False,
            verbose=True,
        )
    except Exception as e:
        logger.error(f'Caught exception in multicast run! {e}')
        raise
    logger.info("Finished MultiNest run. Analyzing results")

# Create an Analyzer object
a = analyse.Analyzer(n_params, outputfiles_basename=outputfiles_basename)

# Get the best-fit parameters (highest likelihood point)
bestfit_params = a.get_best_fit()


# Print the best-fit parameters
params_str = ", ".join(
    f"{param}={bestfit_params['parameters'][i]:.8f}"
    for i, param in enumerate(param_names)
)
logger.info(f"Best-fit parameters: {params_str}")
json.dump(bestfit_params["parameters"], open(outputfiles_basename + "params.json", "w"))

# corner plot
logger.info("Making corner plot")
flat_samples = a.get_equal_weighted_posterior()[:, :-1]
medians = np.median(flat_samples, axis=0)
# labels
corner.corner(
    flat_samples,
    labels=param_names,
    show_titles=True,
    truths=medians,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    # add smooth
    smooth=2,
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 12},
    hist_kwargs={"density": True, "alpha": 0.5},
)
# save the figure
corner_plot_file = basedir + "/GRB_04B_mcmc_corner.pdf"
logger.info(f"Saving corner plot: {corner_plot_file}")
plt.savefig(
    corner_plot_file,
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)
corner_plot_file = basedir + "/GRB_04B_mcmc_corner.png"
logger.info(f"Saving corner plot: {corner_plot_file}")
plt.savefig(
    corner_plot_file,
    dpi=300,
    format="png",
    bbox_inches="tight",
)

# light curve fitting plot
params = {}
params['jetType']=args.jetType
params['z']=args.redshift
#params['logepse']=-1
#params['loglf']=np.log10(200.0)
for i, value in enumerate(bestfit_params["parameters"]):
    params[param_names[i]] = value
for key, value in priors_uniform.items():
    if value["low"] == value["high"]:
        params[key] = value["low"]

lc_plot(basedir, params, observed_data=args.fullobsfile)

if args.alert:
    message = f"Sameer: Run is complete. maxllh={maxllh:.2f}. Please check the parameters:" + "\n" + params_str
    Tele_alert(tele_token, chat_id, message)

