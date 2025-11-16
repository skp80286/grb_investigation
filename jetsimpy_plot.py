# jetsimpy_plot.py — Plot GRB afterglow model light curves and overlay observations
#
# Description:
# - Reads observed photometry from a CSV (`--obsfile`) with columns: Filt, Times, Fluxes, FluxErrs.
# - Builds a jetsimpy model from `--params` (dict/JSON), supports log-prefixed keys (e.g., loge0 → e0).
# - Computes multi-band model fluxes, overlays observations, and saves plots and a log in `<obsdir>/output/`.
#
# CLI:
#   --obsfile  Path to observations CSV.
#   --params   JSON/dict of model parameters; include `jetType` (gaussian|powerlaw|tophat) and `z`.
#   --label    Optional label (reserved).
#
# Outputs:
#   <obsdir>/output/lc_afterflow_obs_matching.pdf and .png
#   <obsdir>/output/jetsimpy_plot_.log
#
# Example:
#   python jetsimpy_plot.py --obsfile data/GRB250916A_cons.csv  \
# --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, \
# n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'
# 
# You can also use this code as a library.
# Example:
# import jetsimpy_plot as jsim
# %matplotlib inline # if you want to show plots interactively in a jupyter notebook
# import matplotlib.pyplot as plt

# params={'jetType': 'tophat', 'e0': 4.87e52, 'epsb': 0.0448, 'epse': 0.3981, 'n0': 0.0032, 'thc': 0.0623, 'thv': 0.0014, 'p': 2.3578, 'loglf': 100, 'A': 0, 's': 0, 'z': 2.011}
# jsim.lc_plot(basedir="output", params=params, observed_data='data/GRB250916A_cons.csv', show_plot=True, save_plot=False)

import json
import os
import random
import datetime
import warnings

import sys

import jetsimpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
from scipy import stats
from scipy.optimize import curve_fit, minimize, newton
import logging
import argparse
from jsonargparse import ArgumentParser

######################

def model(obs_time, obs_nu, params):
    #print(f'obs_time={obs_time},\nobs_nu={obs_nu},\nparams={params},\nz={z}')
    dl = cosmo.luminosity_distance(params['z']).to("Mpc").value

    # Generic transform: any key starting with 'log' -> strip 'log' and 10**value
    for k in list(params.keys()):
        if isinstance(k, str) and k.startswith('log'):
            new_k = k[3:]
            params[new_k] = 10 ** params[k]
            params.pop(k)

    # Backward compatibility for exp-prefixed angles
    if "expthc" in params: 
        params['thc'] = np.log10(params['expthc'])
        params.pop('expthc')
    if "expthv" in params: 
        params['thv'] = np.log10(params['expthv'])
        params.pop('expthv')

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
        A=params["A"],  # no wind
        s=params["s"],
    )

    jetProfile = None
    #logging.info(f"Creating jet {P} {jet_P}")
    #logging.info(f"P={P}\njet_P={jet_P}")
    if params['jetType'] == 'gaussian': 
        jetProfile = jetsimpy.Gaussian(
                jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"]
        )
    elif params['jetType'] == 'powerlaw':
        jetProfile = jetsimpy.PowerLaw(
                jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"], s=jet_P["s"]
        )
    else:
        jetProfile = jetsimpy.TopHat(
                jet_P["theta_c"], jet_P["Eiso"], lf0=jet_P["lf"]
        )

    jet = jetsimpy.Jet(
        jetProfile,
        nwind=jet_P["A"],
        nism=jet_P["n0"],
        grid=jetsimpy.ForwardJetRes(jet_P["theta_c"], 129),
        # allow spread or not
        spread=True,
        tmin=1.0,
        tmax=3.2e9,
        tail=True,
        cal_level=1,
        rtol=1e-6,
        cfl=0.9,
    )

    try:
        model_flux = jet.FluxDensity(
            obs_time,  # [second]
            obs_nu,  # [Hz]
            P,
            model="sync",  # radiation model
            rtol=1e-3,  # integration tolerance
            max_iter=100,
            force_return=True,
        )
        #logging.info(f'model_flux={model_flux}')
    except Exception as e:
        #logging.info(f"obs_time={obs_time}")
        raise
    return model_flux

#multipliers = {'X-ray(1keV)': 10.0, 'X-ray(10keV)': 100.0, 'g': 1.0, 'L': 1, 'R': 1,'r': 1, 
#'i': 8.0, 'u': 8.0, 'z': 16.0, 'J': 32.0, 
#'radio(1.3GHz)': 100.0, 'radio(6GHz)': 400, 'radio(10GHz)': 1500, 'radio(15GHz)': 2000}
multipliers = {'X-ray(10keV)': 10.0, 'g': 1.0, 'r': 4, 'i': 8.0, 'z': 16.0, 'J': 32.0, 'radio(10GHz)': 1500}
filt_freqs={'i':393170436721311.5, 'z':328215960148894.2,
    'VT_B':605000000000000.0, 'VT_R':381000000000000.0, 'r':481130569731985.2, 'J':240000000000000.0, 
    'g':628495719077568.1,'R':468671768303359.2, 'L':86898551000000,
    'SAO-R':45562310000000, 'X-ray(10keV)': 2.42e+18, 'X-ray(1keV)': 2.42e+17,
    'radio(1.3GHz)': 1.3e9, 'radio(6GHz)': 6e9, 'radio(10GHz)': 1e10, 'radio(15GHz)': 1.5e10, 'u': 865201898990000}

def lc_plot(basedir, median_params, sig3_params, observed_data, show_plot=False, save_plot=True):

    # Time and Frequencies
    ta = 1.0e4
    tb = 3.0e6
    t = np.geomspace(ta, tb, num=100)

    df_allobs = pd.read_csv(observed_data)
    df_fit = None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # cmap = matplotlib.colormaps.get_cmap('rainbow_r')  # or 'plasma', 'cividis', 'magma'
    # colors = cmap(np.linspace(0, 1, len(filt_freqs)))

    colors=['tab:purple', 'darkgreen', 'tab:red', 'darkgoldenrod', 'olive', 'royalblue', '#580F41', 'lavender', 'orange', 'cyan']

    # plot the model curves - expected lightcurve from jetsimpy
    j = -1
    for i, (band,nu) in enumerate(filt_freqs.items()):
        if band in multipliers: multiplier = multipliers[band]
        else: continue
        j += 1
        #print(f"Calculating for frequency: {nu}")
        Fnu_model = []

        Fnu_model = model(t, [nu], median_params)
        #print(f'Fnu_model: {Fnu_model}')
        Fnu_model = np.array(Fnu_model)
        #print(f'Fnu_model.shape: {Fnu_model.shape}')

        ax.plot(t, Fnu_model*multiplier,  linewidth=1.0, label=f'{band} x {multiplier}', color=colors[j % len(colors)])
        for params in sig3_params:
            Fnu_model = np.array(model(t, [nu], params))
            ax.plot(t, Fnu_model*multiplier,  linewidth=1.0, color=colors[j % len(colors)], alpha=0.1)


    # plot the actual observations
    j = -1
    for i, (band,nu) in enumerate(filt_freqs.items()):
        if band in multipliers: multiplier = multipliers[band]
        else: continue
        j += 1

        Fnu_allobs = df_allobs[df_allobs['Filt']==band][['Times','Fluxes', 'FluxErrs']].sort_values(by='Times').to_numpy()
        print(f'Plotting band={band}, {len(Fnu_allobs)} rows.')

        ax.errorbar(
                Fnu_allobs[:,0], Fnu_allobs[:,1]*multiplier,
                yerr=Fnu_allobs[:,2]*multiplier,
                fmt='o',
                markersize=8, alpha=1,
                color=colors[j % len(colors)], mec='black',
                elinewidth=0.5, capsize=2
        )

    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-9,1e5)
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$F_\nu$ (mJy)')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Create text content with all Z dictionary values
    z_text = ''
    for key, value in median_params.items():
        if key in ['specType', 'z', 'E0']:
            continue
            # Skip function objects, just show the key
            #z_text += f"{key}: {type(value).__name__}\n"
        else:
            z_text += '\n'
            # Format numerical values
            if key.startswith('log'): 
                key = key[3:]
                value = 10**value
            if isinstance(value, (int, float)):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    z_text += f"{key}: {value:.2e}"
                else:
                    z_text += f"{key}: {value:.4f}"
            else:
                z_text += f"{key}: {value}"

    # Add textbox with all Z dictionary values
    ax.text(0.98, 0.02, z_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.5, edgecolor='none'),
            verticalalignment='bottom', horizontalalignment='right', fontsize=10, fontfamily='monospace')

    #marker_text = "*  Observations used for fitting\nx  All observations\nDashed lines show the best fit"
    #ax.text(0.2, 0.02, marker_text, transform=ax.transAxes,
    #        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black"),
    #        verticalalignment='bottom', fontsize=10, fontfamily='monospace')

    ax.legend(edgecolor='none', loc="lower left", ncol=2)
    fig.tight_layout()

    if save_plot:
        logging.info(f"Saving lightcurve fit plot to: {basedir}/lc_afterflow_obs_matching.pdf")
        fig.savefig(f"{basedir}/lc_afterflow_obs_matching.pdf", format='pdf', bbox_inches='tight')
        logging.info(f"Saving lightcurve fit plot to: {basedir}/lc_afterflow_obs_matching.png")
        fig.savefig(f"{basedir}/lc_afterflow_obs_matching.png", format='png', bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)

################################################
def main():
    SAMPLE_USAGE = (
        "Example:\n"
        "  python jetsimpy_plot.py --obsfile data/GRB250916A_cons.csv  "
        " --params '{jetType: tophat, e0: 4.87e52, epsb: 0.0448, epse: 0.3981, "
        " n0: 0.0032, thc: 0.0623, thv: 0.0014, p: 2.3578, lf: 100, A: 0, s: 0, z: 2.011}'\n"
    )

    parser = ArgumentParser(
        description="Plot GRB afterglow model light curves using jetsimpy and overlay observations.",
        epilog=SAMPLE_USAGE,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def _error(message: str):
        parser.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        print("\n" + SAMPLE_USAGE, file=sys.stderr)
        sys.exit(2)

    parser.error = _error
    parser.add_argument('--label', type=str, default='', help='a descriptive text label to identify this run')
    parser.add_argument('--obsfile', type=str, default='multinest_EP/mcmc_df_trunc.csv', help='csv file containing observed light curve')
    # Will accept --params.key=value and build nested dicts
    parser.add_argument('--params', type=dict, default={})
    args = parser.parse_args()

    np.random.seed(12)

    # read obs csv
    file = args.obsfile

    # Set up the output directory and logging
    basedir =  f"output"
    os.makedirs(basedir, exist_ok=True)
    outputfiles_basename = (
        basedir + f"/jetsimpy_plot_"
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
    lc_plot(basedir, args.params, observed_data=args.obsfile)
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

