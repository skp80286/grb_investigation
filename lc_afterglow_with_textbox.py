import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb

def lc_plot(thetaObs=0, E0=5.11E51, z=0.661, thetaCore=0.05, epsilon_e=0.1, epsilon_b=0.0001, observed_data=None):
    # Jet Parameters
    Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet
        'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum

        'thetaObs':    thetaObs,   # Viewing angle in radians
        'E0':          E0, # Isotropic-equivalent energy in erg
        'thetaCore':   thetaCore,    # Half-opening angle in radians
        'thetaWing':   0.4,    # Outer truncation angle
        'n0':          1.0e-3,    # circumburst density in cm^{-3}
        'p':           2.2,    # electron energy distribution index
        'epsilon_e':   epsilon_e,    # epsilon_e
        'epsilon_B':   epsilon_b,   # epsilon_B
        'xi_N':        1.0,    # Fraction of electrons accelerated
        'd_L':         1.36e26, # Luminosity distance in cm
        'z':           z}   # redshift

    # Time and Frequencies
    ta = 1.0e-2 * grb.day2sec
    tb = 1.0e2 * grb.day2sec
    t = np.geomspace(ta, tb, num=100)

    nuR = 6.0e9
    nuO = 1.0e14
    nuX = 1.0e18

    # Calculate!
    print("Calc Radio")
    FnuR = grb.fluxDensity(t, nuR, **Z)
    print("Calc Optical")
    FnuO = grb.fluxDensity(t, nuO, **Z)
    print("Calc X-ray")
    FnuX = grb.fluxDensity(t, nuX, **Z)

    if observed_data is not None:
        # Load the data from CSV file using numpy
        df = pd.read_csv('data/EP250704a_final_extinct_corr.csv')
        optical_obs = df[df['Filt']=='i'][['Times','Fluxes']].to_numpy()
        radio_obs = df[df['Filt']=='radio'][['Times','Fluxes']].to_numpy()
        xray_obs = df[df['Filt']=='swift'][['Times','Fluxes']].to_numpy()
        print(f"Optical Obs data sample: {optical_obs[:5]}")

        # Print information about the loaded data
        print("Observed Data loaded successfully!")
        print(f"Optical observations (i-band): {optical_obs.shape[0]} data points")
        print(f"Radio observations: {radio_obs.shape[0]} data points")
        print(f"X-ray observations (Swift): {xray_obs.shape[0]} data points")

    # Plot!
    print("Plot")

    tday = t * grb.sec2day

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, FnuR, 'b--', label=r'Radio($\nu=6$ GHz)')
    ax.plot(t, FnuO, 'g--', label=r'Optical($\nu=10^{14}$ Hz)')
    ax.plot(t, FnuX, 'r--.', label=r'X-ray($\nu=10^{18}$ Hz)')

    if observed_data is not None:
        ax.plot(radio_obs[:,0], radio_obs[:,1], 'b-', label=r'Obs Radio')
        ax.plot(optical_obs[:,0], optical_obs[:,1], 'g-', label=r'Obs Optical')
        ax.plot(xray_obs[:,0], xray_obs[:,1], 'r-', label=r'Obs X-ray')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ (d)')
    ax.set_ylabel(r'$F_\nu$ (mJy)')
    ax.legend()
    
    # Create text content with all Z dictionary values
    z_text = "Z Parameters:\n"
    for key, value in Z.items():
        if key in ['jetType', 'specType']:
            # Skip function objects, just show the key
            z_text += f"{key}: {type(value).__name__}\n"
        else:
            # Format numerical values
            if isinstance(value, (int, float)):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    z_text += f"{key}: {value:.2e}\n"
                else:
                    z_text += f"{key}: {value:.4f}\n"
            else:
                z_text += f"{key}: {value}\n"
    
    # Add textbox with all Z dictionary values
    ax.text(0.02, 0.98, z_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black"),
            verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    fig.tight_layout()

    print("Saving plots/lc_afterflow_obs_matching.pdf")
    fig.savefig("plots/lc_afterflow_obs_matching.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    lc_plot(thetaCore=0.02, E0=5.11E51, thetaObs=0.05, epsilon_e=0.005, epsilon_b=0.005, 
            observed_data='data/EP250704a_final_extinct_corr.csv') 