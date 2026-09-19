"""
Neutron stars, pulsars and magnetars.
"""

import numpy as np

from simulation_constants import (
    C,
    G,
    SUN_MASS,
    TOV_MASS,
)

from stellar_pop.stellar_evolution.relations import NEUTRON_STAR_RADIUS_M

# Moment of inertia of a 1.4 Msun, 11.5 km neutron star, in kg m^2.
MOMENT_OF_INERTIA = 1.4e38


def compactness(mass_solar, radius_m=NEUTRON_STAR_RADIUS_M):
    """
    2GM/(Rc^2). About 0.35 for a typical neutron star: strongly
    relativistic, but still outside its own horizon.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return 2.0 * G * m / (np.asarray(radius_m) * C * C)


def surface_gravity(mass_solar, radius_m=NEUTRON_STAR_RADIUS_M):
    """
    In m/s^2: around 2e12, two hundred billion times Earth's.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return G * m / np.asarray(radius_m) ** 2


def mean_density(mass_solar, radius_m=NEUTRON_STAR_RADIUS_M):
    """
    kg/m^3. Comparable to an atomic nucleus, ~2.3e17.
    """

    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS
    r = np.asarray(radius_m, dtype=np.float64)

    return m / (4.0 / 3.0 * np.pi * r**3)


def escape_velocity_fraction_c(mass_solar, radius_m=NEUTRON_STAR_RADIUS_M):
    m = np.asarray(mass_solar, dtype=np.float64) * SUN_MASS

    return np.sqrt(2.0 * G * m / np.asarray(radius_m)) / C


def birth_spin_period(rng, count):
    """
    Newborn pulsars spin with periods of tens of milliseconds, inherited
    from the collapsing core's angular momentum.
    """

    return np.abs(rng.normal(0.05, 0.03, count)) + 0.005


# Calibration of the standard dipole-braking relation
#   B_gauss = 3.2e19 * sqrt(P * Pdot),
# expressed for a field in tesla. This is the convention every pulsar
# catalogue uses, and it folds in the canonical 10 km radius, the
# 1e38 kg m^2 moment of inertia and an orthogonal rotator.
_DIPOLE_FIELD_SCALE = 3.2e15


def magnetic_dipole_spindown(period, magnetic_field=1.0e8):
    """
    dP/dt from magnetic dipole braking, in s/s.
    """

    p = np.maximum(np.asarray(period, dtype=np.float64), 1.0e-4)
    b = np.asarray(magnetic_field, dtype=np.float64)

    return (b / _DIPOLE_FIELD_SCALE) ** 2 / p


def inferred_magnetic_field(period, period_derivative):
    """
    Invert the relation above to read a surface field off P and Pdot.
    """

    p = np.maximum(np.asarray(period, dtype=np.float64), 1.0e-4)
    pdot = np.maximum(np.asarray(period_derivative, dtype=np.float64), 0.0)

    return _DIPOLE_FIELD_SCALE * np.sqrt(p * pdot)


def spin_period_at_age(birth_period, age, magnetic_field=1.0e8):
    """
    Integrating dipole braking gives P(t) = sqrt(P0^2 + 2 k t).
    """

    p0 = np.asarray(birth_period, dtype=np.float64)
    t = np.maximum(np.asarray(age, dtype=np.float64), 0.0)

    k = magnetic_dipole_spindown(1.0, magnetic_field=magnetic_field)

    return np.sqrt(p0 * p0 + 2.0 * k * t)


def characteristic_age(period, period_derivative):
    """
    tau = P / (2 Pdot), the standard pulsar age estimator.
    """

    p = np.asarray(period, dtype=np.float64)
    pdot = np.maximum(np.asarray(period_derivative, dtype=np.float64), 1.0e-30)

    return p / (2.0 * pdot)


def spindown_luminosity(period, period_derivative):
    """
    Rotational energy loss rate in watts: what powers a pulsar wind
    nebula like the Crab.
    """

    p = np.maximum(np.asarray(period, dtype=np.float64), 1.0e-4)
    pdot = np.asarray(period_derivative, dtype=np.float64)

    return 4.0 * np.pi**2 * MOMENT_OF_INERTIA * pdot / p**3


def is_magnetar(magnetic_field) -> bool:
    """
    Magnetars carry fields above ~1e10 tesla, strong enough to distort
    atoms and to crack the crust in starquakes.
    """

    return np.asarray(magnetic_field) > 1.0e10


def maximum_spin_period():
    """
    Break-up rotation period: below about 0.6 ms centrifugal force
    would exceed gravity at the equator.
    """

    m = 1.4 * SUN_MASS

    return 2.0 * np.pi * np.sqrt(NEUTRON_STAR_RADIUS_M**3 / (G * m))


def collapses_to_black_hole(mass_solar) -> bool:
    return np.asarray(mass_solar) > TOV_MASS
