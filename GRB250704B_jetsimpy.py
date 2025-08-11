import json
import os
import sys
import warnings

import corner
import jetsimpy
import matplotlib
import requests

matplotlib.use("Agg")
import argparse
import logging

import numpy as np
import pandas as pd
import pymultinest
import pymultinest.analyse as analyse
from astropy.cosmology import Planck15 as cosmo

np.random.seed(12)

#### telegram:
tele_token = os.environ["Tele_GITbot_Token"]
chat_id = os.environ["Tele_Transient_chat_id"]

def Tele_alert(tele_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{tele_token}/sendMessage?chat_id={chat_id}&text={message}"
        print(requests.get(url).json())
    except Exception as e:
        print("Error in sending message : " + str(e))
######


# check points = data, columns (specially sec or day), basedir, outputfiles_basename, jetType

# file setup
file = "/home/growth/Documents/GRB250704B/mcmc_df.csv"
data = pd.read_csv(file)
required_columns = ["Times", "Freqs", "Fluxes", "FluxErrs"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Input CSV file is missing required columns: {required_columns}")
data = data[data["Fluxes"] > 0]  # taking detections only
if len(data) == 0:
    print("len of the input data is zero")

# input data -- check time in days or sec
obs_time = data["Times"]  # time in days
obs_nu = data["Freqs"].apply(lambda x: int(round(float(x))))  # in Hz
obs_flux = data["Fluxes"]  # in mJy
obs_flux_err = data["FluxErrs"]  # in mJy

# multinest setup
label = "powerlaw"
basedir = f"{os.path.dirname(file)}/multinest/"
os.makedirs(basedir, exist_ok=True)
outputfiles_basename = basedir + f"/GRB_04B_mcmc_{label}_"
n_live_points = 2000

# setup for the params
param_names = ["loge0", "logepsb", "logepse", "logn0", "thc", "thv", "p"]
n_params = len(param_names)
z = 0.661

priors_uniform = {
    "loge0": {"low": 47.0, "high": 53.0},
    "logepsb": {"low": -6.0, "high": -1.0},
    "logepse": {"low": -3.0, "high": -0.5},
    "logn0": {"low": -4.0, "high": 0.0},
    "thc": {"low": 0.01, "high": 0.5},  # radians
    "thv": {"low": 0.01, "high": 0.8},  # radians
    "p": {"low": 2.01, "high": 2.9},
}

def log_prior(cube, ndim, nparams):
    for i, name in enumerate(param_names):
        if name == 'thc' or name == 'thv':
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = np.arccos(np.cos(pmin) - cube[i] * (np.cos(pmin) - np.cos(pmax)))
        else:
            pmin = priors_uniform[name]["low"]
            pmax = priors_uniform[name]["high"]
            cube[i] = pmin + (pmax - pmin) * cube[i]  # scale [0,1] to [min,max]


# def log_likelihood(cube, ndim, nparams):
#     # in normal space
#     params = {name: cube[i] for i, name in enumerate(param_names)}
#     model_flux = model(obs_time, obs_nu, params)
#     residuals = np.log(obs_flux / model_flux)
#     llf = -0.5 * np.sum(residuals**2)
#     return llf

def log_likelihood(cube, ndim, nparams):
    # in log space
    params = {name: cube[i] for i, name in enumerate(param_names)}
    model_flux = model(obs_time, obs_nu, params)
    # log_model_flux = np.log(model_flux)
    # log_obs_flux = np.log(obs_flux)
    log_obs_flux_err = obs_flux_err / obs_flux
    residuals = np.log(obs_flux / model_flux) / log_obs_flux_err
    llf = -0.5 * np.sum(residuals**2)
    return llf


def model(obs_time, obs_nu, params, z=z):
    z = 0.661
    dl = cosmo.luminosity_distance(z).to("Mpc").value
    P = dict(
        eps_e=10 ** params["logepse"],
        eps_b=10 ** params["logepsb"],
        p=params["p"],
        theta_v=params["thv"],
        d=dl,
        z=z,
    )
    
    jet_P = dict(
        Eiso=10 ** params["loge0"],
        lf=1e100,  # no coasting phase
        theta_c=params["thc"],
        n0=10 ** params["logn0"],
        A=0,  # no wind
        s=6,
    )

    jet = jetsimpy.Jet(
        jetsimpy.PowerLaw(
            jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"], s=jet_P["s"]
        ),
        nwind=jet_P["A"],
        nism=jet_P["n0"],
        grid=jetsimpy.ForwardJetRes(jet_P["theta_c"], 129),
        # allow spread or not
        spread=True,
        tmin=10.0,
        tmax=3.2e9,
        tail=True,
        cal_level=1,
        rtol=1e-6,
        cfl=0.9,
    )
    
    model_flux = jet.FluxDensity(
        obs_time,  # [second]
        obs_nu,  # [Hz]
        P,
        model="sync",  # radiation model
        rtol=1e-3,  # integration tolerance
        max_iter=100,
        force_return=True,
    )
    return model_flux


# Run Multinest
pymultinest.run(
    log_likelihood,
    log_prior,
    n_params,
    outputfiles_basename=outputfiles_basename,
    n_live_points=n_live_points,
    sampling_efficiency=0.8,
    evidence_tolerance=0.3,
    # importance_nested_sampling=False,
    resume=False,
    verbose=True,
)
# Create an Analyzer object
a = analyse.Analyzer(n_params, outputfiles_basename=outputfiles_basename)

# Get the best-fit parameters (highest likelihood point)
bestfit_params = a.get_best_fit()

# Print the best-fit parameters
params_str = ", ".join(
    f"{param}={bestfit_params['parameters'][i]:.8f}"
    for i, param in enumerate(param_names)
)
json.dump(bestfit_params["parameters"], open(outputfiles_basename + "params.json", "w"))

message = "Run is complete. Please check the parameters:" + "\n" + params_str
Tele_alert(tele_token, chat_id, message)
