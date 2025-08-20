import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import afterglowpy as grb
import pandas as pd

class Observations:
    """
    A class to hold observational data with band, frequency, times, and fluxes.
    """
    def __init__(self, band, frequency, times, fluxes, fluxerrs):
        self.band = band
        self.frequency = frequency
        self.times = times
        self.fluxes = fluxes
        self.fluxerrs = fluxerrs
    
    def __repr__(self):
        return f"Observations(band='{self.band}', frequency={self.frequency:.2e} Hz, n_points={len(self.times)})"

def model(theta, observations):
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_b, E0 = theta
    thetaWing = 0.4
    # Jet Parameters
    Z = {'jetType':    grb.jet.Gaussian,     # TopHat jet
        'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum
        'thetaObs':    thetaObs,   # Viewing angle in radians
        'E0':          E0, # Isotropic-equivalent energy in erg
        'thetaCore':   thetaCore,    # Half-opening angle in radians
        'thetaWing':   thetaWing,    # Outer truncation angle
        'n0':          n0,    # circumburst density in cm^{-3}
        'p':           p,    # electron energy distribution index
        'epsilon_e':   epsilon_e,    # epsilon_e
        'epsilon_B':   epsilon_b,   # epsilon_B
        'xi_N':        1.0,    # Fraction of electrons accelerated
        'd_L':         1.26e28, # Luminosity distance in cm
        'z':           0.661}   # redshift
    
        

    # Fluxes computed by the model
    fluxes = []
    for obs in observations:
        fluxes.append(grb.fluxDensity(obs.times, obs.frequency, **Z))

    return fluxes

def observations(filename):
    """
    Load observational data from CSV file and create a dictionary with band as key and Observations object as value.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with band names as keys and Observations objects as values
    """
    # Load the data from CSV file
    df = pd.read_csv(filename)
    
    # Create dictionary to store Observations objects for each band
    observations_dict = {}
    
    # Get unique filter bands
    unique_filters = df['Filt'].unique()
    
    # Create Observations object for each filter/band
    for filt in unique_filters:
        # Filter data for this specific band
        filter_data = df[df['Filt'] == filt]
        
        # Extract the data
        band = filt
        frequency = filter_data['Freqs'].iloc[0]  # Assuming same frequency for same filter
        times = filter_data['Times'].values
        fluxes = filter_data['Fluxes'].values
        fluxerrs = filter_data['FluxErrs'].values
        
        # Create Observations object
        obs_obj = Observations(band, frequency, times, fluxes, fluxerrs)
        observations_dict[band] = obs_obj
    
    # Print information about the loaded data
    print("Observational Data loaded successfully!")
    for band, obs in observations_dict.items():
        print(f"band:{band}: {obs}")
    
    return list(observations_dict.values())

def log_likelihood(theta, obslist):
    fluxes_model = model(theta, obslist)
    #print(f"Modeled fluxes: \n{fluxes_model}")
    #print(f"Observed fluxes: \n{obslist}")
    error = 0
    for i in range(len(fluxes_model)):
        #print(f"Modeled flux: {fluxes_model[i]}, Observed flux: {obslist[i].fluxes}")
        #print(f"Observed flux error: {obslist[i].fluxerrs}")
        error += np.sum((np.log(fluxes_model[i]) - np.log(obslist[i].fluxes))**2)
    #print(f"Error: {error}")
    #print(f"Error^2: {error**2}")
    return -0.5 * error

def log_prior(theta):
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_b, E0 = theta
    # Hard boundaries
    
    if not (0 <= thetaObs <= 0.4 and
            0 < thetaCore < 0.2 and
            #0.1 <= thetaWing <= 1 and
            1e-4 < n0 < 10 and
            2.1 < p <2.5 and
            1e-3 < epsilon_e < 0.1 and
            1e-3 < epsilon_b < 0.1 and
            1e50 < E0 < 1e53):
        return -np.inf
    """
    if not (0 <= thetaObs <= 0.4 and
            #0.0199999 <= thetaCore <= 0.0200001 and
            0 < thetaCore < 0.2 and
            #0.1 <= thetaWing <= 1 and
            0.0009999 < n0 < 0.00100001 and
            2.199999 < p <2.200001 and
            0.0049999 < epsilon_e < 0.0050001 and
            1e-3 < epsilon_b < 0.1 and
            5.109999e51 < E0 < 5.110001e51):
        return -np.inf
    """
    # Gaussian priors (log of normal PDF without constants)
    lp = 0.0
    # 0 to 0.4
    lp += -0.5 * (thetaObs / 0.02)**2
    # 0 to 0.2
    lp += -0.5 * ((thetaCore - 0.02)/0.01)**2   
    # 0.1 to 0.5
    # lp += -0.5 * ((thetaWing - 0.4)/0.1)**2   
    # -4 o 1
    lp += -0.5 * ((n0 - 0.001)/0.0005)**2       
    # 2.1 to 2.5
    lp += -0.5 * ((p - 2.2)/0.1)**2              
    # -1 to -3
    lp += -0.5 * ((epsilon_e - 0.005)/0.002)**2  
    lp += -0.5 * ((epsilon_b - 0.05)/0.02)**2      
    lp += -0.5 * ((E0 - 5.11e51)/1e51)**2      
    return lp


def log_posterior(theta, obslist):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obslist)

# 'data/EP250704a_final_extinct_corr.csv'

nwalkers = 14
n_steps = 10000
burn_in = 100
thin = 10

#initial_pos = np.random.rand(nwalkers, ndim)
initial_guess = np.array([0.065, 0.02, 0.001, 2.2, 0.005, 0.05, 5.11e51])
ndim = len(initial_guess)
# Define a scale for each parameter (a fraction of its typical value)
scale = np.array([0.01, 0.01, 0.0001, 0.05, 0.001, 0.01, 1e50])
initial_pos = initial_guess + scale * np.random.randn(nwalkers, ndim)
obslist = observations('data/EP250704a_rem_prompt.csv')#final_extinct_corr.csv')
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(obslist,))
sampler.run_mcmc(initial_pos, n_steps, progress=True)
flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
log_prob_flat = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)

# Index of the maximum log-probability
max_prob_index = np.argmax(log_prob_flat)

# Best-fit parameters
best_fit_params = flat_samples[max_prob_index]

paramnames = ['thetaObs', 'thetaCore', 'n0', 'p', 'epsilon_e', 'epsilon_b', 'E0']
# Print them
print("Best-fit parameters:")
for i, param in enumerate(best_fit_params):
    print(f"{paramnames[i]}={param:.8e},")

for i in range(flat_samples.shape[1]):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"Param {i}: {mcmc[1]:.5f} (+{q[1]:.5f}/-{q[0]:.5f})")

fig = corner.corner(flat_samples, labels=paramnames, truths=best_fit_params)
fig.savefig('plots/corner_plot.pdf', format='pdf', bbox_inches='tight')
#plt.show()
