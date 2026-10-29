"""
Prior definitions for GRB jet modeling with multinest.
"""

# Dirty fireball priors
priors_dirty_fireball = {
    # Isotropic equivalent energy [erg] — dirty fireballs can still be energetic
    "loge0": {"low": 50, "high": 54},
    # Magnetic field equipartition fraction — typically low in dirty fireballs
    "logepsb": {"low": -5, "high": -1},
    # Electron equipartition fraction — moderate range
    "logepse": {"low": -2.0, "high": -0.5},
    # ISM number density [cm^-3] — dirty fireballs often in denser environments
    "logn0": {"low": -2.0, "high": 1.0},
    # Half-opening angle of jet core [rad] — broader jets common with low Γ₀
    "logthc": {"low": -1.5, "high": -0.3},  # ~0.03–0.5 rad (~2°–30°)
    # Viewing angle [rad]
    "logthv": {"low": -2.0, "high": 0.0},  # ~0.01–1.0 rad
    # Electron spectral index — steeper spectrum expected in dirty fireballs
    "p": {"low": 2.01, "high": 3.0},
    # Jet structure power-law index (structured jet)
    "s": {"low": 1, "high": 6},
    # log initial Lorentz factor — KEY: dirty fireball is low Γ₀
    "loglf": {"low": 1.0, "high": 3.0},  # Γ₀ ~ 10–100
    # Wind-like medium parameter (if using wind medium A*)
    "logA": {"low": 0, "high": 0},
}

# Structured off-axis priors (example ranges for a structured/off-axis jet)
priors_structured_offaxis = {
    "loge0": {"low": 52, "high": 55},
    "logepsb": {"low": -6, "high": -2},
    "logepse": {"low": -2.5, "high": -0.5},
    "logn0": {"low": -4.0, "high": 1.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 2.5},
    "s": {"low": 2, "high": 8},
    "loglf": {"low": 2.0, "high": 8.0},
    "logA": {"low": 0, "high": 0},
}

# Generic priors for a broad class of GRBs
priors_generic = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 58},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -1.76, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~..–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": 0, "high": 0},
}

# Generic priors for a broad class of GRBs
priors_generic_highb = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 58},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -2.0, "high": -0.5},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -1.76, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.5, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": 0, "high": 0},
}

priors_generic_wind = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 58},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]. Set to 0 for pure wind.
    "n0": {"low": 0.0, "high": 0.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -1.76, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": -10.0, "high": 10.0},  # Wind medium strongly motivated
    # Wind density slope, n ∝ r^{-k_m}
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_generic_ism_plus_wind = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -2.5, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": -10.0, "high": 10.0},  # Wind medium strongly motivated
    # Wind density slope, n ∝ r^{-k_m}
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_onax = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -2.5, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},
    # On axis
    "logksi": {"low": -2.0, "high": -0.01},
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.0},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": 0, "high": 0},
}

# classical tophat long GRB
priors_tophat = {
    "loge0": {"low": 51, "high": 55},
    "logepsb": {"low": -5, "high": -1},
    "logepse": {"low": -2.0, "high": -0.5},
    "logn0": {"low": -4.0, "high": 1.0},
    "logthc": {"low": -1.5, "high": -0.5},  # ~0.03–0.3 rad
    "logthv": {"low": -3.0, "high": -1.0},  # On-axis: thv << thc
    "p": {"low": 2.01, "high": 2.8},
    "s": {"low": 0.0, "high": 0.0},
    "loglf": {"low": 2.0, "high": 4.0},
    "logA": {"low": -10.0, "high": 10.0},
    "k_m": {"low": 1.0, "high": 3.0},
}

# short GRB
priors_sgrb = {
    "loge0": {"low": 49, "high": 53},  # Less energetic than LGRBs
    "logepsb": {"low": -5, "high": -1},
    "logepse": {"low": -2.0, "high": -0.5},
    "logn0": {"low": -6.0, "high": -1.0},  # KEY: very low ISM density
    "logthc": {"low": -2.0, "high": -0.8},  # Narrow jet ~0.01–0.16 rad
    "logthv": {"low": -2.0, "high": 0.0},
    "p": {"low": 2.0, "high": 2.6},
    "s": {"low": 1.0, "high": 6.0},
    "loglf": {"low": 2.0, "high": 4.0},  # High Γ₀; clean fireball
    "logA": {"low": -10.0, "high": 10.0},  # ISM favoured; low wind
    "k_m": {"low": 1.0, "high": 3.0},
}

# Ultra-long GRB
priors_ulgrb = {
    "loge0": {"low": 51, "high": 56},  # High total energy; long engine
    "logepsb": {"low": -4, "high": -1},
    "logepse": {"low": -2.0, "high": -0.5},
    "logn0": {"low": -2.0, "high": 2.0},  # Dense wind from BSG progenitor
    "logthc": {"low": -1.5, "high": -0.3},  # Wide jet; slower, wider outflow
    "logthv": {"low": -3.0, "high": -0.2},  # On-axis: thv << thc
    "p": {"low": 2.0, "high": 3.0},
    "s": {"low": 1.0, "high": 5.0},
    "loglf": {"low": 1.5, "high": 2.5},  # Moderate Γ₀; not ultra-clean
    "logA": {"low": -10.0, "high": 10.0},  # Wind medium strongly motivated
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_collapsar = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -4, "high": -1},
    "logepse": {"low": -2.0, "high": -0.5},
    "logn0": {"low": -1.0, "high": 2.0},  # Dense wind-shaped CSM
    "logthc": {"low": -1.5, "high": -0.5},
    "logthv": {"low": -2.5, "high": -0.5},
    "p": {"low": 2.1, "high": 2.8},
    "s": {"low": 1.0, "high": 6.0},
    "loglf": {"low": 2.0, "high": 2.8},
    "logA": {"low": 0.0, "high": 2.0},  # KEY: strong wind A* ~ 1–100
    "k_m": {"low": 1.0, "high": 3.0},
}

# Low-luminosity GRB (LLGRB)
priors_llgrb = {
    "loge0": {"low": 48, "high": 51},  # KEY: sub-energetic ~10^48–10^51
    "logepsb": {"low": -3, "high": -0.5},  # Higher εB plausible
    "logepse": {"low": -1.5, "high": -0.3},
    "logn0": {"low": -1.0, "high": 2.0},  # Dense local environment
    "logthc": {"low": -1.0, "high": 0.0},  # Wide jet / quasi-spherical
    "logthv": {"low": -1.5, "high": 0.2},  # Can be viewed at wide angles
    "p": {"low": 2.1, "high": 3.2},  # Steeper spectra observed
    "s": {"low": 1.0, "high": 4.0},
    "loglf": {"low": 0.5, "high": 1.5},  # KEY: Γ₀ ~ 3–30; mildly relativistic
    "logA": {"low": -10.0, "high": 10.0},
    "k_m": {"low": 1.0, "high": 3.0},
}

# classical tophat long GRB
priors_230812B = {
    "loge0": {"low": 51, "high": 55},
    "logepsb": {"low": -5, "high": -1},
    "logepse": {"low": -1.5, "high": -0.5},
    "logn0": {"low": -4.0, "high": 1.0},
    "logthc": {"low": -1.5, "high": -0.2},  #
    "logthv": {"low": -3.0, "high": -0.2},  # On-axis: thv << thc
    "p": {"low": 2.6, "high": 2.6},  # Fixed
    "s": {"low": 0.0, "high": 0.0},
    "loglf": {"low": 1, "high": 4},
    "logA": {"low": 0, "high": 0},
}

priors_230812B_wind = {
    "loge0": {"low": 51, "high": 55},
    "logepsb": {"low": -5, "high": -1},
    "logepse": {"low": -1.5, "high": -0.5},
    "logn0": {"low": -3.0, "high": 1.0},
    "logthc": {"low": -1.5, "high": -0.2},  #
    "logthv": {"low": -3.0, "high": -0.2},  # On-axis: thv << thc
    "p": {"low": 2.6, "high": 2.6},  # Fixed
    "s": {"low": 0.0, "high": 0.0},
    "loglf": {"low": 1, "high": 4},
    "logA": {"low": -2, "high": 0.5},
    "k_m": {"low": 2.0, "high": 2.0},
}
#
priors_250916A = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -5.0, "high": -0.5},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.63, "high": 2.63},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_gam2 = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -5.0, "high": -0.5},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 2.0},
    "logA": {"low": 0, "high": 0},
}
priors_250916A_gam3 = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -5.0, "high": -0.5},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_gam5 = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -5.0, "high": -0.5},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 5.0, "high": 5.0},
    "logA": {"low": 0, "high": 0},
}
priors_250916A_Gamma = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -5.0, "high": -0.5},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_onax = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -6.0, "high": -3},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax1 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.3},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 2.8},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax_ksi = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -1.0},  # radians
    "logthv": {"low": -3.0, "high": 0.0},  # radians
    "logksi": {"low": 0.0, "high": 2.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 1.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_onax_ksi = {
    "loge0": {"low": 53, "high": 55},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -1.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -3.0, "high": 0.0},  # radians
    "logksi": {"low": -2.0, "high": 0.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.63, "high": 2.63},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_onax_gam3 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -6.0, "high": -3},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax_gam3 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 3.0, "high": 3.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_onax_gam2 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -6.0, "high": -3},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 2.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax_gam2 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 2.0, "high": 2.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_onax_gam10 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},
    "logthv": {"low": -6.0, "high": -3},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 10.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_250916A_offax_gam10 = {
    "loge0": {"low": 51, "high": 56},
    "logepsb": {"low": -8, "high": -1},
    "logepse": {"low": -2.5, "high": -0.8},
    "logn0": {"low": -2.0, "high": 0.0},
    "logthc": {"low": -3.0, "high": -0.5},  # radians
    "logthv": {"low": -0.5, "high": 0.0},
    "p": {"low": 2.01, "high": 3.0},
    "s": {"low": 1.0, "high": 8.0},
    "loglf": {"low": 10.0, "high": 10.0},
    "logA": {"low": 0, "high": 0},
}

priors_260924A = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 53, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -3.5, "high": -0.2},
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": -0.2},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.2},
    # Jet structure power-law index
    "s": {"low": 6, "high": 6},
    # log initial Lorentz factor
    "loglf": {"low": 2, "high": 2.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": 0, "high": 0},
}

priors_GRB260924A_wind = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 53, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]. Set to 0 for pure wind.
    "n0": {"low": 0.0, "high": 0.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -3.5, "high": -0.2},
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": -0.2},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.2},
    # Jet structure power-law index
    "s": {"low": 6, "high": 6},
    # log initial Lorentz factor
    "loglf": {"low": 2, "high": 2.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": -3.0, "high": 0.5},  # Wind medium strongly motivated
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_GRB260924A_ism_plus_wind = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -1.824, "high": -1.296},
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.1, "high": 2.1},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},  # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": -3.0, "high": 1.0},  # Wind medium strongly motivated
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_ism_fixlf = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -1.0, "high": -1.0},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -2.5, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~..–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 3.0, "high": 3.0},  # Γ₀ ~ 1000
    # Wind-like medium parameter
    "logA": {"low": 0.0, "high": 0.0},
}

priors_ism_plus_wind_fixlf = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -1, "high": -1},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 1.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -2.5, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~..–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 2.0, "high": 2.0},  # Γ₀ ~ 100
    # Wind-like medium parameter
    "logA": {"low": -10.0, "high": 10.0},  # Wind medium strongly motivated
    "k_m": {"low": 1.0, "high": 3.0},
}

priors_wind_fixlf = {
    # Isotropic equivalent energy [erg] — wide range to accommodate various energies
    "loge0": {"low": 49, "high": 58},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]. Set to 0 for pure wind.
    "n0": {"low": 0.0, "high": 0.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -1.76, "high": -0.2},  # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -5.0, "high": 0.497},  # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.5},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 3.0, "high": 3.0},  # Γ₀ ~ 1000
    # Wind-like medium parameter
    "logA": {"low": -10.0, "high": 10.0},  # Wind medium strongly motivated
    # Wind density slope, n ∝ r^{-k_m}
    "k_m": {"low": 1.0, "high": 3.0},
}

# Map of available priors sets
priors_map = {
    "generic": priors_generic,
    "onax": priors_onax,
    "dirty_fireball": priors_dirty_fireball,
    "sgrb": priors_sgrb,
    "ulgrb": priors_ulgrb,
    "collapsar": priors_collapsar,
    "llgrb": priors_llgrb,
    "structured_offaxis": priors_structured_offaxis,
    "tophat": priors_tophat,
    "230812B": priors_230812B,
    "230812B_wind": priors_230812B_wind,
    "250916A": priors_250916A,
    "250916A_Gamma": priors_250916A_Gamma,
    "250916A_onax": priors_250916A_onax,
    "250916A_offax": priors_250916A_offax,
    "250916A_offax1": priors_250916A_offax1,
    "250916A_onax_gam3": priors_250916A_onax_gam3,
    "250916A_offax_gam3": priors_250916A_offax_gam3,
    "250916A_onax_gam2": priors_250916A_onax_gam2,
    "250916A_offax_gam2": priors_250916A_offax_gam2,
    "250916A_onax_gam10": priors_250916A_onax_gam10,
    "250916A_offax_gam10": priors_250916A_offax_gam10,
    "250916A_onax_ksi": priors_250916A_onax_ksi,
    "250916A_offax_ksi": priors_250916A_offax_ksi,
    "250916A_gam2": priors_250916A_gam2,
    "250916A_gam3": priors_250916A_gam3,
    "250916A_gam5": priors_250916A_gam5,
    "260924A": priors_260924A,
    "generic_highb": priors_generic_highb,
    "generic_wind": priors_generic_wind,
    "generic_ism_plus_wind": priors_generic_ism_plus_wind,
    "GRB260924A_wind": priors_GRB260924A_wind,
    "GRB260924A_ism_plus_wind": priors_GRB260924A_ism_plus_wind,
    "ism_plus_wind_fixlf": priors_ism_plus_wind_fixlf,
    "wind_fixlf": priors_wind_fixlf,
    "ism_fixlf": priors_ism_fixlf,
}
