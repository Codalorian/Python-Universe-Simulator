"""
Stellar remnants: white dwarfs, neutron stars and stellar-mass black
holes, plus the cooling tracks that make them fade over cosmic time.
"""

import numpy as np

from simulation_constants import (
    YEAR,
    SUN_MASS,
    SUN_RADIUS,
    C,
    G,
)

from stellar_pop.stellar_evolution.relations import (
    FATE_WHITE_DWARF,
    FATE_NEUTRON_STAR,
    FATE_BLACK_HOLE,
    FATE_DIRECT_COLLAPSE,
    FATE_PAIR_INSTABILITY,
    white_dwarf_mass,
    white_dwarf_radius,
    white_dwarf_temperature,
    neutron_star_mass,
    black_hole_mass,
    schwarzschild_radius,
    NEUTRON_STAR_RADIUS_M,
)

# Neutron stars are born at ~1e11 K but cool through neutrino emission
# within seconds; the observable surface settles near 1e6 K and then
# cools slowly by photon emission.
NEUTRON_STAR_BIRTH_TEMPERATURE = 1.0e6


def remnant_mass(initial_mass_solar, fate):
    """
    Mass left behind, in solar masses. Pair-instability supernovae
    leave nothing at all.
    """

    m = np.asarray(initial_mass_solar, dtype=np.float64)
    f = np.asarray(fate)

    return np.select(
        [
            f == FATE_WHITE_DWARF,
            f == FATE_NEUTRON_STAR,
            f == FATE_BLACK_HOLE,
            f == FATE_DIRECT_COLLAPSE,
            f == FATE_PAIR_INSTABILITY,
        ],
        [
            white_dwarf_mass(m),
            neutron_star_mass(m),
            black_hole_mass(m),
            # Direct collapse keeps most of the star.
            np.maximum(0.7 * m, 3.0),
            np.zeros_like(m),
        ],
        default=np.zeros_like(m),
    )


def remnant_radius_solar(remnant_mass_solar, fate):
    """
    Radius in solar radii. For black holes this is the event horizon,
    which is what the renderer should draw.
    """

    m = np.asarray(remnant_mass_solar, dtype=np.float64)
    f = np.asarray(fate)

    is_bh = (f == FATE_BLACK_HOLE) | (f == FATE_DIRECT_COLLAPSE)

    return np.select(
        [f == FATE_WHITE_DWARF, f == FATE_NEUTRON_STAR, is_bh],
        [
            white_dwarf_radius(np.maximum(m, 0.05)),
            np.full_like(m, NEUTRON_STAR_RADIUS_M / SUN_RADIUS),
            schwarzschild_radius(np.maximum(m, 1.0e-6)) / SUN_RADIUS,
        ],
        default=np.full_like(m, 1.0e-9),
    )


def neutron_star_temperature(age_since_formation):
    """
    Photon-cooling track, T ~ t^-1/4 after the first ~1000 years of
    neutrino cooling.
    """

    t = np.maximum(np.asarray(age_since_formation, dtype=np.float64), 1.0 * YEAR)

    reference = 1_000.0 * YEAR

    return np.maximum(
        NEUTRON_STAR_BIRTH_TEMPERATURE * (t / reference) ** -0.25,
        1.0e4,
    )


def remnant_temperature(age_since_formation, remnant_mass_solar, fate):
    """
    Surface temperature in kelvin. Black holes get the Hawking
    temperature, which for stellar masses is ~1e-8 K: utterly dark.
    """

    age = np.asarray(age_since_formation, dtype=np.float64)
    m = np.asarray(remnant_mass_solar, dtype=np.float64)
    f = np.asarray(fate)

    is_bh = (f == FATE_BLACK_HOLE) | (f == FATE_DIRECT_COLLAPSE)

    return np.select(
        [f == FATE_WHITE_DWARF, f == FATE_NEUTRON_STAR, is_bh],
        [
            white_dwarf_temperature(age, np.maximum(m, 0.05)),
            neutron_star_temperature(age),
            hawking_temperature(np.maximum(m, 1.0e-6)),
        ],
        default=np.zeros_like(m),
    )


def remnant_luminosity_solar(radius_solar, temperature_kelvin):
    """
    L / Lsun = (R / Rsun)^2 (T / Tsun)^4.
    """

    from simulation_constants import SUN_TEMPERATURE

    r = np.asarray(radius_solar, dtype=np.float64)
    t = np.asarray(temperature_kelvin, dtype=np.float64)

    return (r * r) * (t / SUN_TEMPERATURE) ** 4


def hawking_temperature(mass_solar):
    """
    T = hbar c^3 / (8 pi G M k_B), in kelvin.

    About 6e-8 K for a solar mass: colder than the CMB by eight orders
    of magnitude, so astrophysical black holes gain mass rather than
    evaporate.
    """

    from simulation_constants import HBAR, K_B

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), 1.0e-12) * SUN_MASS

    return HBAR * C**3 / (8.0 * np.pi * G * m * K_B)


def hawking_evaporation_time(mass_solar):
    """
    Evaporation timescale in seconds, t ~ 5120 pi G^2 M^3 / (hbar c^4).

    For a solar mass this is 2e67 years.
    """

    from simulation_constants import HBAR

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), 1.0e-12) * SUN_MASS

    return 5120.0 * np.pi * G**2 * m**3 / (HBAR * C**4)


def is_remnant(fate_state_code) -> bool:
    return fate_state_code in (
        FATE_WHITE_DWARF,
        FATE_NEUTRON_STAR,
        FATE_BLACK_HOLE,
        FATE_DIRECT_COLLAPSE,
    )
