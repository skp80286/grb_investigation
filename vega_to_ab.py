#!/usr/bin/env python3

"""
Convert Vega magnitude to AB magnitude using astropy.

Usage (CLI):
    python vega_to_ab.py J 16.9

Or import as a function:
    from vega_to_ab import vega_to_ab
    m_ab = vega_to_ab("J", 16.9)
"""

import sys
import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody
from astropy.constants import h, c

# ------------------------------------------------------
# Central wavelengths for common filters
# (approximate, can be refined per instrument)
# ------------------------------------------------------
FILTER_WAVELENGTH = {
    "U": 365 * u.nm,
    "B": 445 * u.nm,
    "V": 551 * u.nm,
    "R": 658 * u.nm,
    "I": 806 * u.nm,
    "J": 1.235 * u.um,
    "H": 1.662 * u.um,
    "K": 2.159 * u.um,
    "g": 477 * u.nm,
    "r": 623 * u.nm,
    "i": 763 * u.nm,
    "z": 905 * u.nm,
}

# ------------------------------------------------------
# Vega reference spectrum approximation
# (A0V star ~ 9600 K blackbody)
# BlackBody evaluates to intensity (surface brightness) in erg/(s cm^2 Hz sr).
# Multiply by Vega's solid angle to get flux density at Earth.
# Vega angular diameter ~ 3.24 mas -> Omega ~ 1.93e-16 sr
# ------------------------------------------------------
vega_model = BlackBody(temperature=9600 * u.K)
VEGA_ANGULAR_DIAMETER = 3.24 * u.mas  # milliarcsec
# Solid angle of a disk: Omega = pi * (angular_radius)^2 (rad^2 -> sr)
VEGA_SOLID_ANGLE = (np.pi * ((VEGA_ANGULAR_DIAMETER / 2).to(u.rad)) ** 2).to(u.sr)


def vega_to_ab(filter_name, m_vega):
    """
    Convert Vega magnitude to AB magnitude.

    Parameters
    ----------
    filter_name : str
        Filter name (e.g., 'J', 'R', 'g')
    m_vega : float
        Vega magnitude

    Returns
    -------
    float
        AB magnitude
    """

    if filter_name not in FILTER_WAVELENGTH:
        raise ValueError(f"Unknown filter: {filter_name}")

    wavelength = FILTER_WAVELENGTH[filter_name]

    # Vega intensity (surface brightness) at that wavelength: erg/(s cm^2 Hz sr)
    i_vega = vega_model(wavelength)
    # Flux density at Earth = intensity * solid angle
    f_vega = (i_vega * VEGA_SOLID_ANGLE).to(u.erg / (u.s * u.cm**2 * u.Hz))
    # Convert cgs to Jy (1 Jy = 1e-23 erg/(s cm^2 Hz))
    fnu_vega = f_vega.to(u.Jy)

    # Vega magnitude → flux
    flux = fnu_vega * 10 ** (-0.4 * m_vega)

    # AB magnitude definition
    m_ab = -2.5 * np.log10((flux / (3631 * u.Jy)).to(u.dimensionless_unscaled))

    return float(m_ab)


# ------------------------------------------------------
# CLI interface
# ------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vega_to_ab.py FILTER MAG")
        sys.exit(1)

    filt = sys.argv[1]
    mag = float(sys.argv[2])

    print(vega_to_ab(filt, mag))
