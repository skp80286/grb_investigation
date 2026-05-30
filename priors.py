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
    "logthc": {"low": -1.5, "high": -0.3},   # ~0.03–0.5 rad (~2°–30°)
    # Viewing angle [rad]
    "logthv": {"low": -2.0, "high": 0.0},    # ~0.01–1.0 rad
    # Electron spectral index — steeper spectrum expected in dirty fireballs
    "p": {"low": 2.01, "high": 3.0},
    # Jet structure power-law index (structured jet)
    "s": {"low": 1, "high": 6},
    # log initial Lorentz factor — KEY: dirty fireball is low Γ₀
    "loglf": {"low": 1.0, "high": 3.0},      # Γ₀ ~ 10–100
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
    "loge0": {"low": 49, "high": 56},
    # Magnetic field equipartition fraction
    "logepsb": {"low": -6, "high": -1},
    # Electron equipartition fraction
    "logepse": {"low": -2.5, "high": -0.3},
    # ISM number density [cm^-3]
    "logn0": {"low": -3.0, "high": 2.0},
    # Half-opening angle of jet core [rad]
    "logthc": {"low": -2.5, "high": -0.2},   # ~0.003–0.6 rad
    # Viewing angle [rad]
    "logthv": {"low": -2.5, "high": 0.5},    # ~0.003–3 rad
    # Electron spectral index
    "p": {"low": 2.01, "high": 3.0},
    # Jet structure power-law index
    "s": {"low": 1, "high": 8},
    # log initial Lorentz factor
    "loglf": {"low": 1.0, "high": 4.0},      # Γ₀ ~ 10–10,000
    # Wind-like medium parameter
    "logA": {"low": 0, "high": 0},
}

# Map of available priors sets
priors_map = {
    "generic": priors_generic,
    "dirty_fireball": priors_dirty_fireball,
    "structured_offaxis": priors_structured_offaxis,
}
