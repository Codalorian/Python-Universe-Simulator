"""
Orbital stability and giant impacts.

Packed planetary systems are only metastable. Mutual Hill separation
decides how long they last; below about ten Hill radii, orbits cross
within the lifetime of the star and planets collide.
"""

import numpy as np

from simulation_constants import (
    G,
    AU,
    YEAR,
    EARTH_MASS,
    SUN_MASS,
)


def mutual_hill_radius(a1_m, a2_m, m1_kg, m2_kg, star_mass_kg):
    """
    R_H,mutual = ((m1 + m2) / 3M)^(1/3) * (a1 + a2) / 2
    """

    a1 = np.asarray(a1_m, dtype=np.float64)
    a2 = np.asarray(a2_m, dtype=np.float64)

    return ((np.asarray(m1_kg) + np.asarray(m2_kg)) / (3.0 * np.asarray(star_mass_kg))) ** (
        1.0 / 3.0
    ) * (a1 + a2) * 0.5


def hill_separation(a1_m, a2_m, m1_kg, m2_kg, star_mass_kg):
    """
    Separation of two orbits measured in mutual Hill radii. Below ~10
    the pair is dynamically unstable; the Solar System's neighbours sit
    at 20 to 40.
    """

    r_h = mutual_hill_radius(a1_m, a2_m, m1_kg, m2_kg, star_mass_kg)

    return np.abs(np.asarray(a2_m) - np.asarray(a1_m)) / np.maximum(r_h, 1.0)


def instability_timescale(separation_hill_radii):
    """
    Empirical scaling from N-body surveys (Chambers et al. 1996):

        log10(t / yr) ~ -1.0 + 0.4 * Delta

    Ten Hill radii gives a few thousand years; forty gives longer than
    the age of the universe.
    """

    delta = np.asarray(separation_hill_radii, dtype=np.float64)

    return 10.0 ** np.clip(-1.0 + 0.4 * delta, 0.0, 30.0) * YEAR


def is_stable(a1_m, a2_m, m1_kg, m2_kg, star_mass_kg, over_time):
    """
    Whether a pair survives for a given duration.
    """

    delta = hill_separation(a1_m, a2_m, m1_kg, m2_kg, star_mass_kg)

    return instability_timescale(delta) > over_time


def system_stability(system, over_time):
    """
    Check every adjacent pair in a PlanetarySystem.

    Returns a list of (inner_index, outer_index, separation, survives).
    """

    planets = system.planets

    if len(planets) < 2:
        return []

    star_mass_kg = system.star_mass_solar * SUN_MASS

    results = []

    for i in range(len(planets) - 1):
        inner = planets[i]
        outer = planets[i + 1]

        delta = float(
            hill_separation(
                inner.semi_major_axis_au * AU,
                outer.semi_major_axis_au * AU,
                inner.mass_earth * EARTH_MASS,
                outer.mass_earth * EARTH_MASS,
                star_mass_kg,
            )
        )

        results.append((i, i + 1, delta, instability_timescale(delta) > over_time))

    return results


def impact_energy(m1_kg, m2_kg, relative_velocity_m_s):
    """
    Kinetic energy of a collision in the centre-of-mass frame, in
    joules.
    """

    m1 = np.asarray(m1_kg, dtype=np.float64)
    m2 = np.asarray(m2_kg, dtype=np.float64)

    reduced = m1 * m2 / np.maximum(m1 + m2, 1.0)

    return 0.5 * reduced * np.asarray(relative_velocity_m_s) ** 2


def catastrophic_disruption_threshold(mass_kg, radius_m):
    """
    Energy needed to disperse half a body against its own gravity, in
    joules. Above this the target is destroyed rather than cratered.
    """

    m = np.asarray(mass_kg, dtype=np.float64)
    r = np.maximum(np.asarray(radius_m, dtype=np.float64), 1.0)

    return 0.3 * G * m * m / r


def collision_outcome(m1_kg, m2_kg, r1_m, r2_m, relative_velocity_m_s):
    """
    Classify a giant impact.

    Returns one of "merge", "erosion", or "disruption".
    """

    energy = impact_energy(m1_kg, m2_kg, relative_velocity_m_s)

    threshold = catastrophic_disruption_threshold(
        np.asarray(m1_kg) + np.asarray(m2_kg), np.maximum(r1_m, r2_m)
    )

    ratio = energy / np.maximum(threshold, 1.0)

    return np.where(ratio < 0.5, "merge", np.where(ratio < 2.0, "erosion", "disruption"))


def moon_forming_impact(m_target_kg, m_impactor_kg):
    """
    Mass of the debris disc left in orbit after a grazing giant impact,
    in kg. The Moon formed this way, from roughly a Mars-sized
    impactor on the young Earth.
    """

    return 0.02 * (np.asarray(m_target_kg) + np.asarray(m_impactor_kg))
