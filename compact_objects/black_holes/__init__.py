"""
Black holes, from stellar-mass remnants up to the supermassive holes
at the centres of galaxies.
"""

import numpy as np

from simulation_constants import (
    C,
    G,
    SUN_MASS,
    SIGMA_THOMSON,
    M_PROTON,
)

from stellar_pop.stellar_evolution.relations import (
    isco_radius,
)


def gravitational_radius(mass_solar):
    """
    r_g = GM/c^2, in metres. Half the Schwarzschild radius.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return G * m / (C * C)


def photon_sphere_radius(mass_solar):
    """
    The unstable circular photon orbit at 3 GM/c^2, which is what sets
    the apparent size of a black hole's shadow.
    """

    return 3.0 * gravitational_radius(mass_solar)


def shadow_radius(mass_solar):
    """
    Apparent radius of the shadow to a distant observer: sqrt(27) GM/c^2,
    about 2.6 Schwarzschild radii.
    """

    return np.sqrt(27.0) * gravitational_radius(mass_solar)


def tidal_disruption_radius(black_hole_mass_solar, star_mass_solar, star_radius_solar):
    """
    Distance at which a star is torn apart by tidal forces, in metres.

    Inside the horizon for holes above ~1e8 Msun, which is why the most
    massive black holes swallow stars whole with no flare.
    """

    from simulation_constants import SUN_RADIUS

    m_bh = np.asarray(black_hole_mass_solar, dtype=np.float64)
    m_star = np.maximum(np.asarray(star_mass_solar, dtype=np.float64), 1.0e-6)
    r_star = np.asarray(star_radius_solar, dtype=np.float64) * SUN_RADIUS

    return r_star * (m_bh / m_star) ** (1.0 / 3.0)


def eddington_luminosity_watts(mass_solar):
    """
    L_Edd = 4 pi G M m_p c / sigma_T.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return 4.0 * np.pi * G * m * M_PROTON * C / SIGMA_THOMSON


def eddington_accretion_rate(mass_solar, efficiency=0.1):
    """
    Accretion rate that produces the Eddington luminosity, in kg/s.
    """

    return eddington_luminosity_watts(mass_solar) / (efficiency * C * C)


def accretion_luminosity_watts(accretion_rate, efficiency=0.1):
    """
    L = eta * Mdot * c^2. A standard thin disc around a Schwarzschild
    hole converts about 6% of rest mass; a maximally spinning Kerr hole
    reaches 42%, making accretion the most efficient sustained energy
    source in the universe.
    """

    return np.asarray(accretion_rate, dtype=np.float64) * efficiency * C * C


def radiative_efficiency(spin):
    """
    Binding energy at the ISCO, which is the fraction of infalling rest
    mass released as radiation.
    """

    a = np.clip(np.asarray(spin, dtype=np.float64), 0.0, 0.998)

    r_isco = isco_radius(1.0, a) / gravitational_radius(1.0)

    return 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * r_isco))


def m_sigma_relation(velocity_dispersion_km_s):
    """
    Supermassive black hole mass from the bulge velocity dispersion
    (Gultekin et al. 2009):

        log10(M/Msun) = 8.12 + 4.24 log10(sigma / 200 km/s)
    """

    sigma = np.maximum(np.asarray(velocity_dispersion_km_s, dtype=np.float64), 1.0)

    return 10.0 ** (8.12 + 4.24 * np.log10(sigma / 200.0))


def smbh_mass_from_bulge(bulge_mass_solar):
    """
    Magorrian relation: the central black hole carries roughly 0.2% of
    the stellar bulge mass.
    """

    m = np.maximum(np.asarray(bulge_mass_solar, dtype=np.float64), 1.0)

    return np.clip(0.002 * m, 1.0e3, 7.0e10)


def sphere_of_influence(mass_solar, velocity_dispersion_km_s):
    """
    Radius inside which the hole dominates the gravity, in metres.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS
    sigma = np.maximum(np.asarray(velocity_dispersion_km_s, dtype=np.float64), 1.0) * 1.0e3

    return G * m / (sigma * sigma)


def salpeter_time(efficiency=0.1):
    """
    e-folding time for a black hole growing at the Eddington limit:
    about 45 million years. This is the clock that limits how fast
    quasars can grow.
    """

    return efficiency * SIGMA_THOMSON * C / (4.0 * np.pi * G * M_PROTON * (1.0 - efficiency))


def grow_at_eddington(mass_solar, duration, eddington_ratio=1.0, efficiency=0.1):
    """
    Exponential growth: M(t) = M0 exp(lambda t / t_Salpeter).
    """

    t_s = salpeter_time(efficiency)

    m = np.asarray(mass_solar, dtype=np.float64)

    return m * np.exp(np.asarray(eddington_ratio) * np.asarray(duration) / t_s)


def merger_final_spin(spin_a=0.0, spin_b=0.0, mass_ratio=1.0):
    """
    Approximate final spin of a binary black hole merger. Equal-mass
    non-spinning holes leave a remnant with a ~= 0.686, a robust result
    from numerical relativity.
    """

    q = np.clip(np.asarray(mass_ratio, dtype=np.float64), 0.0, 1.0)
    eta = q / (1.0 + q) ** 2

    a_total = (np.asarray(spin_a) + np.asarray(spin_b) * q * q) / (1.0 + q) ** 2

    return np.clip(a_total + eta * (2.0 * np.sqrt(3.0) - 3.5171 * eta + 2.5763 * eta * eta), 0.0, 0.998)


def merger_radiated_fraction(mass_ratio=1.0):
    """
    Fraction of the total mass radiated as gravitational waves: about
    5% for an equal-mass merger.
    """

    q = np.clip(np.asarray(mass_ratio, dtype=np.float64), 0.0, 1.0)
    eta = q / (1.0 + q) ** 2

    return 0.05 * eta / 0.25


def hole_class(mass_solar) -> str:
    m = float(mass_solar)

    if m < 100.0:
        return "stellar-mass black hole"
    if m < 1.0e5:
        return "intermediate-mass black hole"
    if m < 1.0e9:
        return "supermassive black hole"

    return "ultramassive black hole"
