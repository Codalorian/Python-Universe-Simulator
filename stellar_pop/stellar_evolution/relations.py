"""
Vectorised main-sequence and remnant structure relations.

Every function here takes NumPy arrays and returns NumPy arrays. The
simulation holds its stellar population as a struct-of-arrays and calls
these on the whole population at once, which is what keeps a
quarter-million evolving stars inside a single 60 Hz frame.

Units are solar unless a name says otherwise.
"""

import numpy as np

from simulation_constants import (
    YEAR,
    SUN_MASS,
    SUN_RADIUS,
    SUN_LUMINOSITY,
    SUN_TEMPERATURE,
    G,
    C,
    CHANDRASEKHAR_MASS,
    TOV_MASS,
    MIN_STAR_MASS_SOLAR,
    CORE_COLLAPSE_MIN_MASS,
    DIRECT_COLLAPSE_MIN_MASS,
    PAIR_INSTABILITY_MIN_MASS,
    PAIR_INSTABILITY_MAX_MASS,
)

# Fraction of a star's mass that passes through the hydrogen-burning
# core over its main-sequence life. This is U-shaped in mass: low-mass
# stars are fully convective and can eventually burn nearly all of
# their hydrogen, solar-type stars have a small radiative core, and
# massive stars regain a large convective core.
_CORE_FRACTION_SOLAR = 0.14
_CORE_FRACTION_MIN = 0.08
_CORE_FRACTION_MAX = 0.60

# Hydrogen burning converts 0.7% of rest mass to radiation.
_HYDROGEN_EFFICIENCY = 0.007


def main_sequence_luminosity(mass_solar):
    """
    Broken-power-law mass-luminosity relation, in solar luminosities.

    The breakpoints are the standard ones: the fully convective regime
    below 0.43 Msun, the L ~ M^4 regime up to 2 Msun, L ~ M^3.5 for
    intermediate masses, and the Eddington-limited near-linear regime
    for the most massive stars.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)

    # The most massive stars approach their Eddington limit, which
    # flattens the relation to L ~ M^2.2.
    return np.select(
        [m < 0.43, m < 2.0, m < 20.0],
        [0.23 * m**2.3, m**4.0, 1.4 * m**3.5],
        default=_L_AT_20 * (m / 20.0) ** 2.2,
    )


_L_AT_20 = 1.4 * 20.0**3.5


def main_sequence_radius(mass_solar):
    """
    Main-sequence radius in solar radii.

    R ~ M^0.8 below a solar mass, R ~ M^0.57 above it.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)

    return np.where(m < 1.0, m**0.8, m**0.57)


def effective_temperature(luminosity_solar, radius_solar):
    """
    Stefan-Boltzmann inversion: T = Tsun * (L / R^2)^(1/4).

    Deriving T rather than fitting it keeps L, R and T mutually
    consistent, which matters once stars swell into giants.
    """

    lum = np.maximum(np.asarray(luminosity_solar, dtype=np.float64), 1.0e-12)
    rad = np.maximum(np.asarray(radius_solar, dtype=np.float64), 1.0e-9)

    return SUN_TEMPERATURE * (lum / (rad * rad)) ** 0.25


def core_fraction(mass_solar):
    """
    Fraction of the star's mass that is processed through the core.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)

    fraction = _CORE_FRACTION_SOLAR * np.where(m < 1.0, m**-0.5, m**0.2)

    return np.clip(fraction, _CORE_FRACTION_MIN, _CORE_FRACTION_MAX)


def main_sequence_lifetime(mass_solar, metallicity=0.02):
    """
    Nuclear timescale: t = E_available / L.

    E_available = f_core * X * 0.007 * M c^2, and L is the
    mass-luminosity relation above. This reproduces the familiar
    10 Gyr for the Sun and a few million years for an O star without
    any fitted exponent.

    Metal-poor stars are hotter and more luminous at fixed mass, so
    they burn out somewhat faster; the weak correction below captures
    that trend.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)
    lum = main_sequence_luminosity(m)

    hydrogen_fraction = 0.70

    energy = (
        core_fraction(m)
        * hydrogen_fraction
        * _HYDROGEN_EFFICIENCY
        * m
        * SUN_MASS
        * C
        * C
    )

    lifetime = energy / (lum * SUN_LUMINOSITY)

    z = np.clip(np.asarray(metallicity, dtype=np.float64), 1.0e-5, 0.05)
    metal_factor = (z / 0.02) ** 0.09

    return lifetime * metal_factor


def giant_phase_duration(mass_solar, ms_lifetime):
    """
    Post-main-sequence lifetime: roughly a tenth of the main sequence
    for low-mass stars, shrinking for massive ones whose cores burn
    heavy elements at a runaway pace.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)

    return ms_lifetime * np.where(m < 2.0, 0.12, 0.10 * m ** (-0.3))


def giant_radius(mass_solar, phase_fraction):
    """
    Radius during the giant phase, in solar radii.

    The envelope inflates by up to two orders of magnitude as the core
    contracts. phase_fraction runs 0..1 across the post-main-sequence.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)
    f = np.clip(np.asarray(phase_fraction, dtype=np.float64), 0.0, 1.0)

    base = main_sequence_radius(m)

    # Peak inflation: ~150x for a solar-type star, less for massive ones
    # which are already large.
    peak = np.where(m < 2.0, 160.0, 60.0 * m ** (-0.3))

    return base * (1.0 + (peak - 1.0) * f**2)


def giant_luminosity(mass_solar, phase_fraction):
    """
    Shell burning is far more luminous than core burning, so the star
    brightens by one to two orders of magnitude on the giant branch.
    """

    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), MIN_STAR_MASS_SOLAR)
    f = np.clip(np.asarray(phase_fraction, dtype=np.float64), 0.0, 1.0)

    base = main_sequence_luminosity(m)
    peak = np.where(m < 2.0, 1_500.0, 25.0)

    return base * (1.0 + (peak - 1.0) * f**1.5)


# ------------------------------------------------------------
# Remnants
# ------------------------------------------------------------

def white_dwarf_mass(initial_mass_solar):
    """
    Initial-final mass relation, Kalirai et al. (2008):

        M_final = 0.109 * M_initial + 0.394
    """

    m = np.asarray(initial_mass_solar, dtype=np.float64)

    return np.clip(0.109 * m + 0.394, 0.17, CHANDRASEKHAR_MASS)


def white_dwarf_radius(mass_solar):
    """
    Radius of a non-relativistic electron-degenerate sphere,
    R ~ M^-1/3, with the relativistic correction that drives R to zero
    at the Chandrasekhar mass.
    """

    m = np.clip(np.asarray(mass_solar, dtype=np.float64), 0.05, CHANDRASEKHAR_MASS * 0.999)

    r_earth_units = 0.0126 * m ** (-1.0 / 3.0)

    relativistic = np.sqrt(np.maximum(1.0 - (m / CHANDRASEKHAR_MASS) ** (4.0 / 3.0), 1e-4))

    return r_earth_units * relativistic


def white_dwarf_temperature(age_since_formation, mass_solar):
    """
    Mestel cooling: L ~ t^-7/5, hence T ~ t^-7/20 at fixed radius.

    Newly formed white dwarfs are ~100,000 K; after 10 Gyr they have
    faded to a few thousand.
    """

    t = np.maximum(np.asarray(age_since_formation, dtype=np.float64), 1.0e3 * YEAR)

    reference_age = 1.0e6 * YEAR
    reference_temperature = 100_000.0

    return np.maximum(
        reference_temperature * (t / reference_age) ** (-0.35),
        2_000.0,
    )


def neutron_star_mass(initial_mass_solar):
    """
    Remnant mass from a core-collapse supernova. The iron core mass
    grows slowly with progenitor mass and is capped by the TOV limit.
    """

    m = np.asarray(initial_mass_solar, dtype=np.float64)

    return np.clip(1.17 + 0.09 * (m - 8.0), 1.17, TOV_MASS)


NEUTRON_STAR_RADIUS_M = 11_500.0


def black_hole_mass(initial_mass_solar):
    """
    Fallback-driven black hole mass. Massive stars shed a large part of
    their envelope to winds before collapse, so the remnant is well
    below the initial mass, and metal-rich stars lose more.
    """

    m = np.asarray(initial_mass_solar, dtype=np.float64)

    return np.maximum(TOV_MASS, 0.35 * m)


def schwarzschild_radius(mass_solar):
    """
    In metres.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return 2.0 * G * m / (C * C)


def isco_radius(mass_solar, spin=0.0):
    """
    Innermost stable circular orbit, in metres.

    6 GM/c^2 for a Schwarzschild hole, shrinking towards 1 GM/c^2 for a
    maximally spinning prograde Kerr hole (Bardeen, Press & Teukolsky).
    """

    a = np.clip(np.asarray(spin, dtype=np.float64), 0.0, 0.998)
    rg = schwarzschild_radius(mass_solar) * 0.5

    z1 = 1.0 + np.cbrt(1.0 - a * a) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)

    return rg * (3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))


# ------------------------------------------------------------
# Fate classification
# ------------------------------------------------------------

# Integer fate codes, kept as small ints so the whole population can be
# classified with one vectorised np.select.
FATE_WHITE_DWARF = 0
FATE_NEUTRON_STAR = 1
FATE_BLACK_HOLE = 2
FATE_DIRECT_COLLAPSE = 3
FATE_PAIR_INSTABILITY = 4

FATE_NAMES = {
    FATE_WHITE_DWARF: "white dwarf",
    FATE_NEUTRON_STAR: "neutron star (Type II supernova)",
    FATE_BLACK_HOLE: "black hole (fallback supernova)",
    FATE_DIRECT_COLLAPSE: "black hole (direct collapse)",
    FATE_PAIR_INSTABILITY: "pair-instability supernova (no remnant)",
}


def stellar_fate(mass_solar):
    """
    Classify how each star ends, from its zero-age main-sequence mass.
    """

    m = np.asarray(mass_solar, dtype=np.float64)

    return np.select(
        [
            m < CORE_COLLAPSE_MIN_MASS,
            m < 25.0,
            m < DIRECT_COLLAPSE_MIN_MASS,
            m < PAIR_INSTABILITY_MIN_MASS,
            m <= PAIR_INSTABILITY_MAX_MASS,
        ],
        [
            FATE_WHITE_DWARF,
            FATE_NEUTRON_STAR,
            FATE_BLACK_HOLE,
            FATE_DIRECT_COLLAPSE,
            FATE_PAIR_INSTABILITY,
        ],
        # Above the pair-instability window, photodisintegration wins
        # and the whole star falls in.
        default=FATE_DIRECT_COLLAPSE,
    ).astype(np.int8)


def habitable_zone_au(luminosity_solar):
    """
    Inner and outer edges of the liquid-water zone, in AU, scaled from
    the solar case as sqrt(L).
    """

    lum = np.maximum(np.asarray(luminosity_solar, dtype=np.float64), 1.0e-9)
    root = np.sqrt(lum)

    return 0.95 * root, 1.67 * root


def surface_gravity(mass_solar, radius_solar):
    """
    In m/s^2.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS
    r = np.maximum(np.asarray(radius_solar, dtype=np.float64), 1.0e-9) * SUN_RADIUS

    return G * m / (r * r)


def eddington_luminosity(mass_solar):
    """
    In solar luminosities: the luminosity at which radiation pressure
    on ionised hydrogen balances gravity.
    """

    m = np.asarray(mass_solar, dtype=np.float64)

    return 3.2e4 * m
