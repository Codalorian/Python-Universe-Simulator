"""
Quasars: supermassive black holes accreting near their Eddington limit.

A quasar is the most luminous persistent object in the universe, and
the reason is a simple energy argument: accretion onto a black hole
converts 6-40% of infalling rest mass into radiation, where hydrogen
fusion manages 0.7%.

The quasar population peaks at z ~ 2 and has faded by two orders of
magnitude since, because the gas supply ran out. That history is
modelled here as an epoch-dependent duty cycle.
"""

import numpy as np

from simulation_constants import (
    C,
    G,
    SIGMA_SB,
    SUN_MASS,
    YEAR,
    MYR,
    PARSEC,
)

from compact_objects.black_holes import (
    eddington_luminosity_watts,
    gravitational_radius,
    radiative_efficiency,
)

from stellar_pop.stellar_evolution.relations import isco_radius


def duty_cycle(z):
    """
    Fraction of supermassive black holes that are actively accreting at
    a given redshift.

    Near zero today, rising to a few percent at cosmic noon. This is
    what makes quasars a young universe phenomenon.
    """

    zz = np.maximum(np.asarray(z, dtype=np.float64), 0.0)

    # Lognormal in (1+z), peaking at z ~ 2.2.
    peak = np.log(3.2)
    width = 0.55

    return 0.06 * np.exp(-0.5 * ((np.log1p(zz) - peak) / width) ** 2) + 0.0005


def eddington_ratio_distribution(rng, count, z):
    """
    Draw L / L_Edd for active nuclei.

    Roughly lognormal, centred near the Eddington limit at high
    redshift and an order of magnitude below it today.
    """

    zz = np.asarray(z, dtype=np.float64)

    centre = np.log(0.03) + 1.2 * np.log1p(np.maximum(zz, 0.0))

    return np.clip(np.exp(rng.normal(centre, 0.8, count)), 1.0e-5, 2.0)


def bolometric_luminosity(mass_solar, eddington_ratio):
    """
    In watts.
    """

    return np.asarray(eddington_ratio, dtype=np.float64) * eddington_luminosity_watts(
        mass_solar
    )


def accretion_rate(mass_solar, eddington_ratio, spin=0.0):
    """
    Mdot in kg/s, from L = eta Mdot c^2.
    """

    eta = radiative_efficiency(spin)

    return bolometric_luminosity(mass_solar, eddington_ratio) / (eta * C * C)


def accretion_rate_solar_per_year(mass_solar, eddington_ratio, spin=0.0):
    return accretion_rate(mass_solar, eddington_ratio, spin) * YEAR / SUN_MASS


def disc_temperature(radius_m, mass_solar, accretion_rate_kg_s, spin=0.0):
    """
    Shakura-Sunyaev thin-disc temperature profile:

        T(r) = [ 3 G M Mdot / (8 pi sigma r^3) * (1 - sqrt(r_in/r)) ]^1/4

    The inner disc of a stellar-mass hole reaches 1e7 K and shines in
    X-rays; a 1e9 Msun quasar disc peaks near 1e5 K in the ultraviolet,
    because the temperature falls as M^-1/4.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS
    mdot = np.asarray(accretion_rate_kg_s, dtype=np.float64)

    r_in = isco_radius(mass_solar, spin)
    r = np.maximum(np.asarray(radius_m, dtype=np.float64), r_in * 1.0001)

    flux = (
        3.0 * G * m * mdot / (8.0 * np.pi * SIGMA_SB * r**3) * (1.0 - np.sqrt(r_in / r))
    )

    return np.maximum(flux, 0.0) ** 0.25


def peak_disc_temperature(mass_solar, eddington_ratio, spin=0.0):
    """
    The maximum disc temperature, reached at ~1.36 r_isco.
    """

    mdot = accretion_rate(mass_solar, eddington_ratio, spin)
    r_in = isco_radius(mass_solar, spin)

    return disc_temperature(1.36 * r_in, mass_solar, mdot, spin)


def disc_outer_radius(mass_solar):
    """
    Self-gravity breaks the disc up beyond ~1e5 gravitational radii.
    """

    return 1.0e5 * gravitational_radius(mass_solar)


def broad_line_region_radius(luminosity_watts):
    """
    Reverberation-mapped size of the broad-line region, in metres.

    The radius-luminosity relation R ~ L^0.5 gives about one parsec
    for a 1e40 W quasar, which is what makes single-epoch black hole
    masses possible: measure the line width, read off the radius, and
    apply the virial theorem.
    """

    lum = np.maximum(np.asarray(luminosity_watts, dtype=np.float64), 1.0)

    return PARSEC * np.sqrt(lum / 1.0e40)


def jet_power(mass_solar, eddington_ratio, spin=0.9):
    """
    Blandford-Znajek jet power: the hole's rotational energy tapped by
    magnetic field lines threading the horizon.

    Scales as a^2, so only rapidly spinning holes launch strong jets.
    """

    a = np.clip(np.asarray(spin, dtype=np.float64), 0.0, 0.998)

    return 0.5 * a * a * bolometric_luminosity(mass_solar, eddington_ratio)


def is_quasar(luminosity_watts) -> bool:
    """
    The conventional dividing line between a Seyfert nucleus and a
    quasar: M_B = -23, about 1e38 W bolometric.
    """

    return np.asarray(luminosity_watts) > 1.0e38


def agn_class(luminosity_watts) -> str:
    lum = float(luminosity_watts)

    if lum < 1.0e34:
        return "quiescent nucleus"
    if lum < 1.0e37:
        return "low-luminosity AGN"
    if lum < 1.0e38:
        return "Seyfert galaxy"
    if lum < 1.0e40:
        return "quasar"

    return "hyperluminous quasar"


def lifetime():
    """
    Typical duration of a single quasar episode: ~100 Myr, comparable
    to a couple of Salpeter e-folding times.
    """

    return 100.0 * MYR


def sample_population(rng, black_hole_masses, cosmic_time, expansion, spins=None):
    """
    Decide which black holes are active right now and how bright they
    are.

    Returns (active_mask, luminosity_watts).
    """

    masses = np.asarray(black_hole_masses, dtype=np.float64)

    z = float(expansion.redshift(cosmic_time))

    active = rng.random(masses.shape) < duty_cycle(z)

    ratios = eddington_ratio_distribution(rng, masses.shape[0], z)

    if spins is None:
        spins = 0.7

    luminosity = np.where(active, bolometric_luminosity(masses, ratios), 0.0)

    return active, luminosity
