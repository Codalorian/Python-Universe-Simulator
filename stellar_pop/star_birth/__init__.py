"""
Star formation: the initial mass function and the cosmic star
formation history.

The IMF is sampled by exact inverse-CDF transform rather than by
rejection, so drawing a million masses is a single vectorised pass.
"""

import numpy as np

from simulation_constants import (
    MIN_STAR_MASS_SOLAR,
    MAX_STAR_MASS_SOLAR,
    MPC,
    YEAR,
    GYR,
)

# ------------------------------------------------------------
# Kroupa (2001) initial mass function
# ------------------------------------------------------------
#
#   dN/dM ~ M^-alpha, with
#       alpha = 1.3   for 0.08 <= M < 0.5
#       alpha = 2.3   for 0.50 <= M

_BREAK = 0.5

_ALPHA_LOW = 1.3
_ALPHA_HIGH = 2.3


def _segment_integral(lo, hi, alpha):
    """
    Integral of M^-alpha from lo to hi.
    """

    power = 1.0 - alpha

    return (hi**power - lo**power) / power


def imf_normalisation(m_min=MIN_STAR_MASS_SOLAR, m_max=MAX_STAR_MASS_SOLAR):
    """
    Returns (weight_low, weight_high, continuity_factor).

    The high-mass segment is scaled so dN/dM is continuous at the
    break, then both are normalised to unit total probability.
    """

    # Continuity: A_low * B^-1.3 == A_high * B^-2.3, so A_high = A_low * B.
    continuity = _BREAK ** (_ALPHA_HIGH - _ALPHA_LOW)

    low = _segment_integral(m_min, _BREAK, _ALPHA_LOW)
    high = continuity * _segment_integral(_BREAK, m_max, _ALPHA_HIGH)

    total = low + high

    return low / total, high / total, continuity


def sample_imf(count, rng, m_min=MIN_STAR_MASS_SOLAR, m_max=MAX_STAR_MASS_SOLAR):
    """
    Draw `count` stellar masses from the Kroupa IMF, in solar masses.

    Inverse-CDF sampling of a two-segment power law: analytic, exact,
    and fully vectorised.
    """

    weight_low, _, _ = imf_normalisation(m_min, m_max)

    u = rng.random(count)

    in_low = u < weight_low

    masses = np.empty(count, dtype=np.float64)

    # Low segment: invert the CDF of M^-1.3 on [m_min, break].
    if in_low.any():
        v = u[in_low] / weight_low
        p = 1.0 - _ALPHA_LOW
        masses[in_low] = (m_min**p + v * (_BREAK**p - m_min**p)) ** (1.0 / p)

    # High segment: invert the CDF of M^-2.3 on [break, m_max].
    high = ~in_low
    if high.any():
        v = (u[high] - weight_low) / (1.0 - weight_low)
        p = 1.0 - _ALPHA_HIGH
        masses[high] = (_BREAK**p + v * (m_max**p - _BREAK**p)) ** (1.0 / p)

    return np.clip(masses, m_min, m_max)


def imf_pdf(mass_solar, m_min=MIN_STAR_MASS_SOLAR, m_max=MAX_STAR_MASS_SOLAR):
    """
    Probability density of the Kroupa IMF, normalised over the allowed
    mass range.

    Needed for importance sampling: when a population is deliberately
    drawn to over-represent rare luminous stars, this is the numerator
    of the weight that puts the statistics back where they belong.
    """

    m = np.clip(np.asarray(mass_solar, dtype=np.float64), m_min, m_max)

    continuity = _BREAK ** (_ALPHA_HIGH - _ALPHA_LOW)

    total = _segment_integral(m_min, _BREAK, _ALPHA_LOW) + continuity * _segment_integral(
        _BREAK, m_max, _ALPHA_HIGH
    )

    return np.where(m < _BREAK, m**-_ALPHA_LOW, continuity * m**-_ALPHA_HIGH) / total


def log_uniform_pdf(mass_solar, m_min, m_max):
    """
    Density of a distribution that is uniform in log mass.
    """

    m = np.asarray(mass_solar, dtype=np.float64)

    inside = (m >= m_min) & (m <= m_max)

    return np.where(inside, 1.0 / (m * np.log(m_max / m_min)), 0.0)


def imf_mean_mass(m_min=MIN_STAR_MASS_SOLAR, m_max=MAX_STAR_MASS_SOLAR):
    """
    Mass-weighted mean of the IMF, ~0.36 Msun for the standard range.
    """

    continuity = _BREAK ** (_ALPHA_HIGH - _ALPHA_LOW)

    number = _segment_integral(m_min, _BREAK, _ALPHA_LOW) + continuity * _segment_integral(
        _BREAK, m_max, _ALPHA_HIGH
    )

    mass = _segment_integral(m_min, _BREAK, _ALPHA_LOW - 1.0) + continuity * _segment_integral(
        _BREAK, m_max, _ALPHA_HIGH - 1.0
    )

    return mass / number


# ------------------------------------------------------------
# Cosmic star formation history
# ------------------------------------------------------------

def madau_dickinson_sfr(z):
    """
    Cosmic star formation rate density in Msun / yr / Mpc^3.

    Madau & Dickinson (2014), equation 15:

        psi(z) = 0.015 (1+z)^2.7 / ( 1 + ((1+z)/2.9)^5.6 )

    Peaks around z = 1.9, "cosmic noon", at ten times the present-day
    rate.
    """

    zz = np.maximum(np.asarray(z, dtype=np.float64), -0.99)

    return 0.015 * (1.0 + zz) ** 2.7 / (1.0 + ((1.0 + zz) / 2.9) ** 5.6)


def sfr_density_si(z):
    """
    Same quantity in kg / s / m^3.
    """

    from simulation_constants import SUN_MASS

    return madau_dickinson_sfr(z) * SUN_MASS / YEAR / MPC**3


def formation_time_cdf(expansion, samples=2048):
    """
    Build a lookup table mapping a uniform deviate to a star formation
    time, weighted by the Madau-Dickinson rate.

    Returns (cdf, times). The star field draws formation times from
    this, which is why the simulated population is dominated by stars
    born around cosmic noon rather than spread uniformly in time.
    """

    t_first = 0.18 * GYR
    t_end = expansion.present_age

    times = np.linspace(t_first, t_end, samples)

    z = expansion.redshift(times)
    rate = madau_dickinson_sfr(z)

    # The comoving volume element is constant, so the number of stars
    # formed per unit cosmic time is just the rate.
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(times))))
    cdf /= cdf[-1]

    return cdf, times


def sample_formation_times(count, rng, expansion, cdf=None, times=None):
    """
    Draw cosmic formation times following the cosmic SFR history.
    """

    if cdf is None or times is None:
        cdf, times = formation_time_cdf(expansion)

    return np.interp(rng.random(count), cdf, times)


# ------------------------------------------------------------
# Chemical enrichment
# ------------------------------------------------------------

def mean_metallicity(cosmic_time, expansion):
    """
    Mass fraction of elements heavier than helium in gas forming stars
    at a given cosmic time.

    Rises from primordial (effectively zero for Population III) to
    roughly solar today, tracking the integrated star formation that
    preceded it.
    """

    z = np.asarray(expansion.redshift(cosmic_time), dtype=np.float64)

    # A simple closed-box-like decline with redshift; [Fe/H] falls by
    # about 0.25 dex per unit redshift out to cosmic noon.
    log_z_over_solar = -0.25 * np.maximum(z, 0.0)

    return 0.0142 * 10.0**log_z_over_solar


def sample_metallicity(formation_times, rng, expansion):
    """
    Per-star metallicity: the epoch mean, scattered by ~0.25 dex to
    represent the spread between and within galaxies.
    """

    mean = mean_metallicity(formation_times, expansion)

    scatter = rng.normal(0.0, 0.25, size=np.shape(formation_times))

    return np.clip(mean * 10.0**scatter, 1.0e-6, 0.05)


def is_population_iii(metallicity):
    """
    Metal-free stars: the first generation, born from pristine
    hydrogen and helium.
    """

    return np.asarray(metallicity) < 1.0e-5
