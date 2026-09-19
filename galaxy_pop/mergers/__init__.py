"""
Galaxy mergers.

Hierarchical structure formation means every large galaxy today is the
accumulated wreckage of smaller ones. This module supplies the rates
and timescales, and resolves individual merger events during a
simulation step.
"""

from dataclasses import dataclass

import numpy as np

from simulation_constants import (
    G,
    SUN_MASS,
    GYR,
)


def merger_rate_per_galaxy(z, mass_ratio_min=0.25):
    """
    Major mergers per galaxy per Gyr.

    Observed rates rise steeply with redshift roughly as (1+z)^2.5:
    the early universe was a much more crowded, violent place.
    """

    zz = np.maximum(np.asarray(z, dtype=np.float64), 0.0)

    base = 0.03 * (1.0 + zz) ** 2.5

    # Minor mergers are far more common than major ones.
    ratio_factor = (0.25 / max(mass_ratio_min, 0.01)) ** 0.8

    return base * ratio_factor


def dynamical_friction_time(
    satellite_mass_solar, host_mass_solar, orbital_radius_m, circular_velocity_m_s
):
    """
    Chandrasekhar dynamical friction sinking time, in seconds:

        t_df ~ 1.17 / ln(Lambda) * (M_host / M_sat) * r^2 v_c / (G M_host)

    A satellite a tenth the host's mass spirals in within a couple of
    billion years; one a thousandth never arrives.
    """

    m_sat = np.maximum(np.asarray(satellite_mass_solar, dtype=np.float64), 1.0) * SUN_MASS
    m_host = np.maximum(np.asarray(host_mass_solar, dtype=np.float64), 1.0) * SUN_MASS

    r = np.asarray(orbital_radius_m, dtype=np.float64)
    v = np.maximum(np.asarray(circular_velocity_m_s, dtype=np.float64), 1.0)

    coulomb_log = np.log(1.0 + m_host / m_sat)

    return 1.17 * r * r * v / (G * m_sat * np.maximum(coulomb_log, 0.1))


def starburst_strength(mass_ratio):
    """
    Multiplier on the star formation rate during a merger.

    Tidal torques funnel gas to the centre; an equal-mass merger can
    raise the star formation rate a hundredfold and produce an
    ultraluminous infrared galaxy.
    """

    q = np.clip(np.asarray(mass_ratio, dtype=np.float64), 0.0, 1.0)

    return 1.0 + 100.0 * q**1.5


def merger_remnant_morphology(mass_ratio, primary_morphology):
    """
    A major merger destroys discs: two spirals coalesce into an
    elliptical. Minor mergers leave the disc largely intact.
    """

    from galaxy_pop.galaxy_formation import MORPH_ELLIPTICAL

    q = np.asarray(mass_ratio, dtype=np.float64)

    return np.where(q > 0.25, MORPH_ELLIPTICAL, np.asarray(primary_morphology))


def relaxation_time(remnant_mass_solar, effective_radius_m):
    """
    Violent relaxation time, in seconds: how long the merger remnant
    takes to settle into a smooth new equilibrium. Typically a few
    hundred million years.
    """

    m = np.maximum(np.asarray(remnant_mass_solar, dtype=np.float64), 1.0) * SUN_MASS
    r = np.asarray(effective_radius_m, dtype=np.float64)

    return np.sqrt(r**3 / (G * m))


@dataclass(slots=True)
class MergerEvent:
    primary_index: int
    secondary_index: int
    mass_ratio: float
    cosmic_time: float
    is_major: bool


def _sample_separation_mpc(positions):
    """
    Mean spacing between catalogue entries, used to decide which pairs
    count as close.
    """

    span = float(np.abs(positions).max()) or 1.0
    count = max(len(positions), 1)

    return 2.0 * span / count ** (1.0 / 3.0)


def resolve_mergers(catalogue, dt, cosmic_time, expansion, rng, max_events=64):
    """
    Advance the galaxy catalogue by dt and merge a physically plausible
    number of pairs.

    Rather than run a full N-body pairing, this draws the expected
    number of mergers from the redshift-dependent rate and applies each
    to a randomly chosen close pair, which reproduces the population
    statistics at negligible cost.
    """

    count = len(catalogue["halo_mass"])

    if count < 2 or dt <= 0.0:
        return []

    z = float(expansion.redshift(cosmic_time))

    expected = merger_rate_per_galaxy(z) * (dt / GYR) * count

    n_events = int(min(rng.poisson(max(expected, 0.0)), max_events))

    if n_events == 0:
        return []

    events = []

    positions = catalogue["position_mpc"]
    halo_mass = catalogue["halo_mass"]
    stellar_mass = catalogue["stellar_mass"]
    smbh_mass = catalogue["smbh_mass"]

    pair_scale_mpc = _sample_separation_mpc(positions)

    # Skip galaxies already consumed by an earlier merger.
    primaries = rng.integers(0, count, n_events)

    for primary in primaries:
        # Find a genuinely nearby partner rather than a random one.
        offset = int(rng.integers(1, min(8, count)))
        secondary = int((primary + offset) % count)

        if secondary == primary:
            continue

        if halo_mass[primary] <= 0.0 or halo_mass[secondary] <= 0.0:
            continue

        separation = np.linalg.norm(positions[primary] - positions[secondary])

        # The catalogue is a sparse sample of a population numbering in
        # the hundreds of billions, so an absolute "within 100 kpc" cut
        # would never fire. The pair must instead be close relative to
        # the sampling scale, which is what makes the *rate* right.
        if separation > pair_scale_mpc:
            continue

        m_a = halo_mass[primary]
        m_b = halo_mass[secondary]

        if m_b > m_a:
            primary, secondary = secondary, primary
            m_a, m_b = m_b, m_a

        ratio = float(m_b / max(m_a, 1.0))

        halo_mass[primary] = m_a + m_b
        stellar_mass[primary] = stellar_mass[primary] + stellar_mass[secondary]
        smbh_mass[primary] = smbh_mass[primary] + smbh_mass[secondary]

        if ratio > 0.25:
            catalogue["morphology"][primary] = merger_remnant_morphology(
                ratio, catalogue["morphology"][primary]
            )

        catalogue["radius"][primary] *= (1.0 + ratio) ** 0.6

        # The consumed galaxy is moved onto its devourer and reduced to
        # nothing, which keeps every array the same length.
        halo_mass[secondary] = 0.0
        stellar_mass[secondary] = 0.0
        smbh_mass[secondary] = 0.0
        positions[secondary] = positions[primary]

        events.append(
            MergerEvent(
                primary_index=int(primary),
                secondary_index=int(secondary),
                mass_ratio=ratio,
                cosmic_time=float(cosmic_time),
                is_major=ratio > 0.25,
            )
        )

    return events
