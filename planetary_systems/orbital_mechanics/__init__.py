"""
Two-body orbital mechanics.

Positions come from solving Kepler's equation

    M = E - e sin E

by Newton-Raphson, vectorised over every planet in a system at once.
That is what lets the simulator run orbits at a billion years per
second and still have the planets be in the right place.
"""

import numpy as np

from simulation_constants import G, C


def orbital_period(semi_major_axis_m, total_mass_kg):
    """
    Kepler's third law: T = 2 pi sqrt(a^3 / GM).
    """

    a = np.maximum(np.asarray(semi_major_axis_m, dtype=np.float64), 1.0)

    return 2.0 * np.pi * np.sqrt(a**3 / (G * np.asarray(total_mass_kg)))


def mean_motion(semi_major_axis_m, total_mass_kg):
    """
    Angular rate in rad/s.
    """

    a = np.maximum(np.asarray(semi_major_axis_m, dtype=np.float64), 1.0)

    return np.sqrt(G * np.asarray(total_mass_kg) / a**3)


def solve_kepler(mean_anomaly, eccentricity, iterations=5):
    """
    Eccentric anomaly E from mean anomaly M.

    The starting guess E0 = M + e sin M is accurate enough that five
    Newton steps converge to machine precision for e < 0.9.
    """

    m = np.asarray(mean_anomaly, dtype=np.float64)
    e = np.clip(np.asarray(eccentricity, dtype=np.float64), 0.0, 0.95)

    # Wrap to [-pi, pi] so the iteration stays well conditioned.
    m = np.mod(m + np.pi, 2.0 * np.pi) - np.pi

    ecc_anomaly = m + e * np.sin(m)

    for _ in range(iterations):
        f = ecc_anomaly - e * np.sin(ecc_anomaly) - m
        df = 1.0 - e * np.cos(ecc_anomaly)

        ecc_anomaly = ecc_anomaly - f / df

    return ecc_anomaly


def true_anomaly(eccentric_anomaly, eccentricity):
    e = np.asarray(eccentricity, dtype=np.float64)
    ea = np.asarray(eccentric_anomaly, dtype=np.float64)

    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(ea * 0.5),
        np.sqrt(1.0 - e) * np.cos(ea * 0.5),
    )


def orbital_radius(semi_major_axis_m, eccentricity, eccentric_anomaly):
    a = np.asarray(semi_major_axis_m, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)

    return a * (1.0 - e * np.cos(eccentric_anomaly))


def position_at_time(
    time_seconds,
    semi_major_axis_m,
    eccentricity,
    inclination,
    longitude_of_node,
    argument_of_periapsis,
    mean_anomaly_at_epoch,
    total_mass_kg,
):
    """
    Full three-dimensional orbital position, in metres, relative to the
    focus. All arguments broadcast, so an entire planetary system is
    one call.

    Returns an (N, 3) array.
    """

    n = mean_motion(semi_major_axis_m, total_mass_kg)

    m = np.asarray(mean_anomaly_at_epoch, dtype=np.float64) + n * np.asarray(
        time_seconds, dtype=np.float64
    )

    ecc_anomaly = solve_kepler(m, eccentricity)

    nu = true_anomaly(ecc_anomaly, eccentricity)
    r = orbital_radius(semi_major_axis_m, eccentricity, ecc_anomaly)

    # Position in the orbital plane.
    x_orbit = r * np.cos(nu)
    y_orbit = r * np.sin(nu)

    cos_w = np.cos(argument_of_periapsis)
    sin_w = np.sin(argument_of_periapsis)
    cos_o = np.cos(longitude_of_node)
    sin_o = np.sin(longitude_of_node)
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)

    # Rotate by argument of periapsis, inclination, then node.
    x = (
        x_orbit * (cos_w * cos_o - sin_w * sin_o * cos_i)
        - y_orbit * (sin_w * cos_o + cos_w * sin_o * cos_i)
    )

    y = (
        x_orbit * (cos_w * sin_o + sin_w * cos_o * cos_i)
        + y_orbit * (cos_w * cos_o * cos_i - sin_w * sin_o)
    )

    z = x_orbit * (sin_w * sin_i) + y_orbit * (cos_w * sin_i)

    return np.stack([x, y, z], axis=-1)


def orbital_velocity(semi_major_axis_m, radius_m, total_mass_kg):
    """
    Vis-viva equation: v^2 = GM (2/r - 1/a).
    """

    a = np.maximum(np.asarray(semi_major_axis_m, dtype=np.float64), 1.0)
    r = np.maximum(np.asarray(radius_m, dtype=np.float64), 1.0)

    return np.sqrt(np.maximum(G * np.asarray(total_mass_kg) * (2.0 / r - 1.0 / a), 0.0))


def escape_velocity(mass_kg, radius_m):
    return np.sqrt(2.0 * G * np.asarray(mass_kg) / np.maximum(np.asarray(radius_m), 1.0))


def hill_radius(semi_major_axis_m, eccentricity, planet_mass_kg, star_mass_kg):
    """
    Radius of a planet's gravitational dominance:

        r_H = a (1 - e) (m / 3M)^(1/3)
    """

    a = np.asarray(semi_major_axis_m, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)

    return a * (1.0 - e) * (np.asarray(planet_mass_kg) / (3.0 * np.asarray(star_mass_kg))) ** (1.0 / 3.0)


def roche_limit(primary_radius_m, primary_density, satellite_density):
    """
    Distance inside which a fluid satellite is torn apart by tides.
    Saturn's rings sit inside Saturn's.
    """

    return (
        2.44
        * np.asarray(primary_radius_m, dtype=np.float64)
        * (np.asarray(primary_density) / np.maximum(np.asarray(satellite_density), 1.0)) ** (1.0 / 3.0)
    )


def perihelion_precession_per_orbit(semi_major_axis_m, eccentricity, star_mass_kg):
    """
    General-relativistic precession, in radians per orbit:

        delta = 6 pi G M / (c^2 a (1 - e^2))

    For Mercury this is the famous 43 arcseconds per century.
    """

    a = np.maximum(np.asarray(semi_major_axis_m, dtype=np.float64), 1.0)
    e = np.asarray(eccentricity, dtype=np.float64)

    return 6.0 * np.pi * G * np.asarray(star_mass_kg) / (C * C * a * (1.0 - e * e))


def tidal_locking_time(
    semi_major_axis_m, planet_mass_kg, planet_radius_m, star_mass_kg, rigidity=3.0e10
):
    """
    Rough timescale for a planet to become tidally locked, in seconds.

    Scales as a^6, which is why Mercury is locked into a spin-orbit
    resonance and Earth is not.
    """

    a = np.asarray(semi_major_axis_m, dtype=np.float64)
    m_p = np.asarray(planet_mass_kg, dtype=np.float64)
    r_p = np.asarray(planet_radius_m, dtype=np.float64)
    m_s = np.asarray(star_mass_kg, dtype=np.float64)

    return 6.0 * a**6 * r_p * rigidity / (G * m_s * m_s * m_p) * 1.0e-2
