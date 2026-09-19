"""
Dark matter haloes.

Every galaxy sits at the centre of a halo that outweighs it roughly
ten to one and sets its rotation curve, its velocity dispersion and
ultimately the mass of its central black hole.

Structure follows the NFW profile:

    rho(r) = rho_s / [ (r/r_s) (1 + r/r_s)^2 ]
"""

import numpy as np

from simulation_constants import (
    G,
    SUN_MASS,
)

# Virial overdensity relative to the mean matter density (Bryan &
# Norman 1998, for a flat LambdaCDM universe).
DELTA_VIRIAL = 200.0


def virial_radius(halo_mass_solar, expansion, cosmic_time):
    """
    Radius enclosing DELTA_VIRIAL times the critical density, in metres.
    """

    m = np.asarray(halo_mass_solar, dtype=np.float64) * SUN_MASS

    h = expansion.hubble_parameter(cosmic_time)
    rho_crit_z = 3.0 * h * h / (8.0 * np.pi * G)

    return (3.0 * m / (4.0 * np.pi * DELTA_VIRIAL * rho_crit_z)) ** (1.0 / 3.0)


def virial_velocity(halo_mass_solar, virial_radius_m):
    """
    Circular velocity at the virial radius, in m/s.
    """

    m = np.asarray(halo_mass_solar, dtype=np.float64) * SUN_MASS

    return np.sqrt(G * m / np.maximum(virial_radius_m, 1.0))


def velocity_dispersion(virial_velocity_m_s):
    """
    sigma ~ v_vir / sqrt(2) for an isothermal sphere.
    """

    return np.asarray(virial_velocity_m_s, dtype=np.float64) / np.sqrt(2.0)


def concentration(halo_mass_solar, redshift):
    """
    Concentration parameter c = r_vir / r_s.

    Low-mass haloes collapsed earlier, when the universe was denser, so
    they are more concentrated. c also falls with redshift as (1+z)^-1.
    """

    m = np.maximum(np.asarray(halo_mass_solar, dtype=np.float64), 1.0e6)
    z = np.maximum(np.asarray(redshift, dtype=np.float64), 0.0)

    return np.clip(9.0 * (m / 1.0e12) ** -0.1 / (1.0 + z), 2.0, 30.0)


def nfw_mass_enclosed(radius_m, halo_mass_solar, virial_radius_m, concentration_value):
    """
    Mass inside a given radius for an NFW halo, in solar masses.
    """

    c = np.asarray(concentration_value, dtype=np.float64)
    rv = np.asarray(virial_radius_m, dtype=np.float64)
    r = np.clip(np.asarray(radius_m, dtype=np.float64), 1.0, None)

    x = np.clip(r / rv, 1.0e-6, None)

    def mu(t):
        return np.log1p(t) - t / (1.0 + t)

    return np.asarray(halo_mass_solar, dtype=np.float64) * mu(c * x) / mu(c)


def nfw_circular_velocity(radius_m, halo_mass_solar, virial_radius_m, concentration_value):
    """
    Rotation curve of the halo, in m/s. Rises, peaks near ~2 r_s and
    then falls slowly, which is what makes observed rotation curves
    flat once the disc is added.
    """

    m_enc = nfw_mass_enclosed(radius_m, halo_mass_solar, virial_radius_m, concentration_value)

    r = np.maximum(np.asarray(radius_m, dtype=np.float64), 1.0)

    return np.sqrt(G * m_enc * SUN_MASS / r)


def nfw_scale_density(halo_mass_solar, virial_radius_m, concentration_value):
    """
    rho_s in kg/m^3.
    """

    c = np.asarray(concentration_value, dtype=np.float64)
    rv = np.asarray(virial_radius_m, dtype=np.float64)

    r_s = rv / c
    mu = np.log1p(c) - c / (1.0 + c)

    m = np.asarray(halo_mass_solar, dtype=np.float64) * SUN_MASS

    return m / (4.0 * np.pi * r_s**3 * mu)


def stellar_mass_from_halo(halo_mass_solar):
    """
    Abundance matching, Moster et al. (2013) double power law:

        M* / M_h = 2 N (x^-beta + x^gamma)^-1,  x = M_h / M_1

    Star formation is most efficient in ~1e12 Msun haloes, where about
    a fifth of the baryons end up in stars. Below that supernovae blow
    the gas out; above it, AGN feedback keeps it too hot to cool.
    """

    m = np.maximum(np.asarray(halo_mass_solar, dtype=np.float64), 1.0e7)

    n = 0.0351
    m1 = 10.0**11.59
    beta = 1.376
    gamma = 0.608

    x = m / m1

    return 2.0 * n * m / (x**-beta + x**gamma)


def halo_mass_function(halo_mass_solar, expansion, cosmic_time):
    """
    Comoving number density of haloes per dex of mass, in Mpc^-3.

    A Press-Schechter-like exponential cutoff above the characteristic
    mass M*, which itself grows as structure forms. This is what makes
    dwarf galaxies overwhelmingly common and clusters rare.
    """

    m = np.maximum(np.asarray(halo_mass_solar, dtype=np.float64), 1.0e7)

    growth = expansion.growth_factor(cosmic_time)

    # Characteristic collapsing mass today, scaled by the growth factor.
    m_star = 1.0e13 * np.asarray(growth) ** 3.0

    x = m / m_star

    return 0.02 * x**-1.0 * np.exp(-(x**0.8))


def sample_halo_masses(count, rng, m_min=1.0e9, m_max=3.0e15):
    """
    Draw halo masses from a Schechter-like distribution by inverse
    sampling of a power law, then rejecting the exponential tail.

    The slope -1.9 in number per unit mass is the standard low-mass
    behaviour of the cold dark matter mass function.
    """

    slope = -1.9
    p = 1.0 + slope

    u = rng.random(count)

    masses = (m_min**p + u * (m_max**p - m_min**p)) ** (1.0 / p)

    # Exponential cutoff above the characteristic cluster mass.
    m_cut = 3.0e14
    keep = rng.random(count) < np.exp(-masses / m_cut)

    # Redraw the rejected ones from the low-mass end rather than
    # looping: the fraction is small and the result is statistically
    # equivalent.
    n_bad = int((~keep).sum())

    if n_bad:
        u2 = rng.random(n_bad)
        masses[~keep] = (m_min**p + u2 * ((m_cut * 0.3) ** p - m_min**p)) ** (1.0 / p)

    return masses


def dynamical_time(halo_mass_solar, virial_radius_m):
    """
    Crossing time of the halo, in seconds. Sets how fast a galaxy can
    respond to anything.
    """

    v = virial_velocity(halo_mass_solar, virial_radius_m)

    return np.asarray(virial_radius_m) / np.maximum(v, 1.0)
