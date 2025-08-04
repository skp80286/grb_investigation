import numpy as np

# Load the data from CSV file using numpy
observed_data = np.genfromtxt('data/EP250704a_final_extinct_corr.csv', 
                             delimiter=',', 
                             skip_header=1, 
                             dtype=None, 
                             encoding='utf-8')

# Extract column indices
# Based on the CSV structure: Times,Filt,mag,magerr,lim_mag,Freqs,Fluxes,FluxErrs,days,Seconds
times_col = 0
filt_col = 1
fluxes_col = 6

# Filter data for each observation type
optical_mask = observed_data[:, filt_col] == b'i'
radio_mask = observed_data[:, filt_col] == b'radio'
xray_mask = observed_data[:, filt_col] == b'swift'

# Create 2D numpy arrays with Times as first column and Fluxes as second column
optical_obs = np.column_stack((observed_data[optical_mask, times_col].astype(float),
                              observed_data[optical_mask, fluxes_col].astype(float)))

radio_obs = np.column_stack((observed_data[radio_mask, times_col].astype(float),
                            observed_data[radio_mask, fluxes_col].astype(float)))

xray_obs = np.column_stack((observed_data[xray_mask, times_col].astype(float),
                           observed_data[xray_mask, fluxes_col].astype(float)))

# Print information about the loaded data
print("Data loaded successfully!")
print(f"Optical observations (i-band): {optical_obs.shape[0]} data points")
print(f"Radio observations: {radio_obs.shape[0]} data points")
print(f"X-ray observations (Swift): {xray_obs.shape[0]} data points")

# Display first few rows of each dataset
print("\nOptical observations (first 5 rows):")
print(optical_obs[:5])

print("\nRadio observations (first 5 rows):")
print(radio_obs[:5])

print("\nX-ray observations (first 5 rows):")
print(xray_obs[:5]) 