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
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
from scipy import stats
from scipy.optimize import curve_fit, minimize, newton
import logging
import argparse
from jsonargparse import ArgumentParser

######################

logger = logging.getLogger(__name__)

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
multipliers = {
    'X-ray(10keV)': 32, 
    'u': 1,
    'g': 2, 
    'VT_B': 4, 
    'r': 8, 
    'R': 16,
    'i': 32, 
    'VT_R': 64, 
    'J': 128, 
    'radio(15.5GHz)': 1024
}

filt_freqs={'i':3.92913E+14, 'z':328215960148894.2,
    'VT_B':5.45077E+14, 'VT_R':3.63385E+14, 'r':4.81208E+14, 'J':2.40161E+14, 
    'g':6.28496E+14,'R':4.67914E+14, 'L':5.55516E+14,
    'SAO-R':45562310000000, 'X-ray(10keV)': 2.42e+18, 'X-ray(1keV)': 2.42e+17,
    'radio(1.3GHz)': 1.3e9, 'radio(6GHz)': 6e9, 'radio(10GHz)': 1e10, 
    'radio(15GHz)': 1.5e10, 'radio(15.5GHz)': 1.55e10,'u': 8.65202E+14}

# cmap = matplotlib.colormaps.get_cmap('rainbow_r')  # or 'plasma', 'cividis', 'magma'
# colors = cmap(np.linspace(0, 1, len(filt_freqs)))

#colors=['tab:purple', 'darkgreen', 'tab:red', 'darkgoldenrod', 'olive', 'royalblue', '#580F41', 'orange', 'cyan']
band_colors={
    'X-ray(10keV)': 'darkviolet', 
    'u': 'teal',
    'VT_B': 'royalblue', 
    'g': 'darkgreen', 
    'r': 'tab:red', 
    'i': 'darkgoldenrod', 
    'VT_R': 'orange', 
    'R': 'magenta',
    'J': 'olive', 
    'radio(15.5GHz)': 'deepskyblue'
}
band_secondary_colors={
    'X-ray(10keV)': 'lavender', 
    'u': 'mediumturquoise',
    'VT_B': 'lightsteelblue', 
    'g': 'mediumaquamarine', 
    'r': 'lightcoral', 
    'i': 'khaki', 
    'VT_R': 'peachpuff', 
    'R': 'plum',
    'J': 'olive', 
    'radio(15.5GHz)': 'lightblue'
}

"""
    band_colors = {
        "radio(15.5GHz)": "#4B0082",  # deep purple
        "L": "#5A0000",               # very dark red (near-IR, long λ)
        "J": "#8B0000",               # dark red / near-IR
        "z": "#A00000",               # deep maroon
        "R": "#E41A1C",               # red
        "VT_R": "#FF7F00",            # orange-red
        "i": "#D95F02",               # amber
        "r": "#4DAF4A",               # green
        "g": "#00A6D6",               # cyan
        "VT_B": "#377EB8",            # blue
        "u": "#984EA3",               # violet / near-UV
        "X-ray(10keV)": "#000000"     # black (extreme high-energy)
    }
"""

def lc_plot(basedir, median_params, sig3_params, observed_data, show_plot=False, save_plot=True):
    plt.style.use(['science', 'high-vis'])

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 5,  # minimum allowed by Nature
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,  # embed fonts as TrueType
        "ps.fonttype": 42,
        # "figure.dpi": 300,  # ensure high-res bitmap export when needed
        "savefig.dpi": 300,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.75,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,

    })
    
    # Time and Frequencies
    ta = 1.0e4
    tb = 3.0e6
    t = np.geomspace(ta, tb, num=100)

    df_allobs = pd.read_csv(observed_data)
    # Convert numeric columns to numeric types, handling invalid values
    df_allobs['Times'] = pd.to_numeric(df_allobs['Times'], errors='coerce')
    df_allobs['Fluxes'] = pd.to_numeric(df_allobs['Fluxes'], errors='coerce')
    df_allobs['FluxErrs'] = pd.to_numeric(df_allobs['FluxErrs'], errors='coerce')
    logger.info(f"lc_plot: len(median_params)={len(median_params)}, len(sig3_parmas)={len(sig3_params)}, len(df_allobs)={len(df_allobs)}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot the model curves - expected lightcurve from jetsimpy
    j = -1
    for i, (band,nu) in enumerate(sorted(filt_freqs.items(), key=lambda x: -x[1])):
        if band in multipliers: multiplier = multipliers[band]
        else: continue
        j += 1
        logger.info(f"Calculating for frequency: {nu}")
        Fnu_model = []

        Fnu_model = model(t, [nu], median_params)
        #logger.info(f'Fnu_model: {Fnu_model}')
        Fnu_model = np.array(Fnu_model)
        #logger.info(f'Fnu_model.shape: {Fnu_model.shape}')

        for params in sig3_params:
            Fnu_model = np.array(model(t, [nu], params))
            ax.plot(t, Fnu_model*multiplier,  linewidth=1.0, linestyle='-', color=band_secondary_colors.get(band, "#000000"), alpha=0.1)
        ax.plot(t, Fnu_model*multiplier,  linewidth=1.0, linestyle='-', label=f'{band} x {multiplier}', color=band_colors.get(band, "#000000"), alpha=1)


    # plot the actual observations
    j = -1
    for i, (band,nu) in enumerate(sorted(filt_freqs.items(), key=lambda x: -x[1])):
        if band in multipliers: multiplier = multipliers[band]
        else: continue
        j += 1

        Fnu_allobs = df_allobs[(df_allobs['Filt']==band) & (df_allobs['UL']=='N')][['Times','Fluxes', 'FluxErrs']].sort_values(by='Times').to_numpy()
        logger.info(f'Plotting band={band}, {len(Fnu_allobs)} rows, err={Fnu_allobs[:,2]}.')

        ax.errorbar(
                Fnu_allobs[:,0], Fnu_allobs[:,1]*multiplier,
                yerr=Fnu_allobs[:,2]*multiplier,
                fmt='o',
                markersize=4, alpha=1,
                color=band_colors.get(band, "#000000"), mec='black',
                elinewidth=0.5, capsize=2
        )

        Fnu_ul_obs = df_allobs[(df_allobs['Filt']==band) & (df_allobs['UL']=='Y')][['Times','Fluxes', 'FluxErrs']].sort_values(by='Times').to_numpy()
        if len(Fnu_ul_obs) > 0:
            logger.info(f'Plotting upper limit band={band}, {len(Fnu_ul_obs)} rows.')

            """
            ax.scatter(
                    Fnu_ul_obs[:,0], Fnu_ul_obs[:,1]*multiplier,
                    marker='v',
                    s=4, alpha=1,
                    c=colors[j % len(colors)], 
                    edgecolors='black', 
            )
            """

            # Plot upper limits with arrows pointing down
            plt.errorbar(
                Fnu_ul_obs[:,0], Fnu_ul_obs[:,1]*multiplier,
                yerr=None,
                fmt='v',
                markersize=6, alpha=1,
                color=band_colors.get(band, "#000000"), mec='black',
                elinewidth=0.5, capsize=2, uplims=True
            )


    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-9,1e5)
    ax.set_xlim(1e4,3e6)
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
            verticalalignment='bottom', horizontalalignment='right', fontsize=12)

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

