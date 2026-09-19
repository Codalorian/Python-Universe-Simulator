"""
Flat LambdaCDM expansion history.

The Friedmann equation

    (a'/a)^2 = H0^2 * ( Omega_r/a^4 + Omega_m/a^3 + Omega_L )

is integrated numerically once at import time to build a monotone
table linking scale factor, cosmic time, redshift, comoving distance
and lookback time.

Every public query is then a vectorised table interpolation, so the
whole module costs a few microseconds per call even when it is handed
a quarter of a million values at once.
"""

import numpy as np

from simulation_constants import (
    C,
    H0,
    OMEGA_M,
    OMEGA_R,
    OMEGA_LAMBDA,
    CMB_TEMPERATURE,
    GYR,
)

# Number of samples in the expansion table. Log-spaced in a, so the
# radiation era and the present day are both resolved well.
_TABLE_SIZE = 4096

_A_MIN = 1.0e-10
_A_MAX = 40.0


def e_function(a):
    """
    Dimensionless expansion rate E(a) = H(a) / H0.
    """

    a = np.asarray(a, dtype=np.float64)
    a = np.maximum(a, _A_MIN)

    return np.sqrt(
        OMEGA_R / a**4
        + OMEGA_M / a**3
        + OMEGA_LAMBDA
    )


def _build_tables():
    """
    Integrate dt/da = 1 / (a H(a)) and dchi/da = c / (a^2 H(a)).

    Both integrands diverge as a -> 0, so the integration is done in
    u = ln(a), where dt/du = 1 / H(a) stays finite and smooth.
    """

    u = np.linspace(np.log(_A_MIN), np.log(_A_MAX), _TABLE_SIZE)
    a = np.exp(u)

    h = H0 * e_function(a)

    # dt/du = 1/H, dchi/du = c / (a H)
    dt_du = 1.0 / h
    dchi_du = C / (a * h)

    # Cumulative trapezoidal integration from a = _A_MIN.
    du = np.diff(u)

    t = np.concatenate(([0.0], np.cumsum(0.5 * (dt_du[1:] + dt_du[:-1]) * du)))
    chi = np.concatenate(([0.0], np.cumsum(0.5 * (dchi_du[1:] + dchi_du[:-1]) * du)))

    # Time below _A_MIN is utterly negligible (< 1e-30 s) but is added
    # analytically anyway: in pure radiation domination t = a^2 / (2 H0 sqrt(Wr)).
    t += _A_MIN**2 / (2.0 * H0 * np.sqrt(OMEGA_R))

    # Comoving distance is measured backwards from today, so store the
    # integral from a to a=1 rather than from the beginning.
    chi = chi[-1] - chi

    return a, t, chi


_A_TABLE, _T_TABLE, _CHI_TABLE = _build_tables()

# Index of a == 1 by interpolation, giving the present age.
PRESENT_AGE = float(np.interp(1.0, _A_TABLE, _T_TABLE))

# Comoving distance offset so that chi(a=1) == 0.
_CHI_TABLE = _CHI_TABLE - float(np.interp(1.0, _A_TABLE, _CHI_TABLE))

# Comoving distance to the Big Bang: the particle horizon today.
PARTICLE_HORIZON = float(_CHI_TABLE[0])

AGE_AT_A_MAX = float(_T_TABLE[-1])


class ExpansionModel:
    """
    Query interface over the precomputed expansion history.

    All methods accept scalars or NumPy arrays and return the same
    shape.
    """

    def __init__(self):
        self.present_age = PRESENT_AGE
        self.particle_horizon = PARTICLE_HORIZON

    # --------------------------------------------------------
    # Scale factor and time
    # --------------------------------------------------------

    def scale_factor(self, cosmic_time):
        """
        a(t). Clamped to the tabulated range at both ends.
        """

        return np.interp(cosmic_time, _T_TABLE, _A_TABLE)

    def cosmic_time(self, scale_factor):
        """
        t(a): the inverse of scale_factor().
        """

        return np.interp(scale_factor, _A_TABLE, _T_TABLE)

    def time_at_redshift(self, z):
        return self.cosmic_time(1.0 / (1.0 + np.asarray(z, dtype=np.float64)))

    def redshift(self, cosmic_time):
        """
        Redshift of light emitted at cosmic_time and seen today.
        """

        return 1.0 / self.scale_factor(cosmic_time) - 1.0

    def lookback_time(self, cosmic_time):
        return PRESENT_AGE - np.asarray(cosmic_time, dtype=np.float64)

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    def hubble_parameter(self, cosmic_time):
        """
        H(t) in SI units (1/s).
        """

        return H0 * e_function(self.scale_factor(cosmic_time))

    def hubble_velocity(self, cosmic_time, proper_distance):
        """
        Recession velocity from the Hubble flow. This exceeds c beyond
        the Hubble radius, which is correct: it is not a local motion.
        """

        return self.hubble_parameter(cosmic_time) * proper_distance

    def hubble_radius(self, cosmic_time):
        return C / self.hubble_parameter(cosmic_time)

    def deceleration_parameter(self, cosmic_time):
        """
        q = Omega_r/a^4 + Omega_m/(2 a^3) - Omega_L, divided by E^2.

        Negative today: the expansion is accelerating.
        """

        a = self.scale_factor(cosmic_time)
        e2 = e_function(a) ** 2

        return (OMEGA_R / a**4 + 0.5 * OMEGA_M / a**3 - OMEGA_LAMBDA) / e2

    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------

    def comoving_distance(self, cosmic_time):
        """
        Comoving distance light has travelled since cosmic_time, in
        metres. Zero at the present day, PARTICLE_HORIZON at t = 0.
        """

        return np.interp(
            self.scale_factor(cosmic_time),
            _A_TABLE,
            _CHI_TABLE,
        )

    def angular_diameter_distance(self, cosmic_time):
        a = self.scale_factor(cosmic_time)
        return self.comoving_distance(cosmic_time) * a

    def luminosity_distance(self, cosmic_time):
        a = self.scale_factor(cosmic_time)
        return self.comoving_distance(cosmic_time) / a

    def comoving_to_proper(self, comoving_distance, cosmic_time):
        return np.asarray(comoving_distance) * self.scale_factor(cosmic_time)

    # --------------------------------------------------------
    # Thermal history
    # --------------------------------------------------------

    def cmb_temperature(self, cosmic_time):
        return CMB_TEMPERATURE / self.scale_factor(cosmic_time)

    def matter_density(self, cosmic_time):
        """
        Mean matter density in kg/m^3.
        """

        from simulation_constants import RHO_CRIT

        return RHO_CRIT * OMEGA_M / self.scale_factor(cosmic_time) ** 3

    # --------------------------------------------------------
    # Growth of structure
    # --------------------------------------------------------

    def growth_factor(self, cosmic_time):
        """
        Linear growth factor D(a), normalised to 1 today.

        Carroll, Press & Turner (1992) fitting formula, accurate to a
        few parts in a thousand for flat LambdaCDM.
        """

        a = np.asarray(self.scale_factor(cosmic_time), dtype=np.float64)
        e2 = e_function(a) ** 2

        om = OMEGA_M / (a**3 * e2)
        ol = OMEGA_LAMBDA / e2

        g = 2.5 * om / (om ** (4.0 / 7.0) - ol + (1.0 + 0.5 * om) * (1.0 + ol / 70.0))

        om0 = OMEGA_M
        ol0 = OMEGA_LAMBDA

        g0 = 2.5 * om0 / (
            om0 ** (4.0 / 7.0) - ol0 + (1.0 + 0.5 * om0) * (1.0 + ol0 / 70.0)
        )

        return a * g / g0


def describe_epoch(cosmic_time):
    """
    Human-readable name of the cosmological epoch at a given time.
    """

    from simulation_constants import (
        T_PLANCK_EPOCH,
        T_INFLATION_END,
        T_QUARK_HADRON,
        T_NUCLEOSYNTHESIS,
        T_MATTER_RADIATION_EQ,
        T_RECOMBINATION,
        T_FIRST_STARS,
        T_REIONIZATION_END,
    )

    t = float(cosmic_time)

    if t <= 0.0:
        return "Singularity"
    if t < T_PLANCK_EPOCH:
        return "Planck epoch"
    if t < T_INFLATION_END:
        return "Inflation"
    if t < T_QUARK_HADRON:
        return "Quark-gluon plasma"
    if t < T_NUCLEOSYNTHESIS:
        return "Hadron / lepton epoch"
    if t < T_MATTER_RADIATION_EQ:
        return "Photon epoch (nucleosynthesis done)"
    if t < T_RECOMBINATION:
        return "Matter-radiation equality"
    if t < T_FIRST_STARS:
        return "Recombination - the Dark Ages"
    if t < T_REIONIZATION_END:
        return "Cosmic dawn / reionisation"
    if t < 3.0 * GYR:
        return "Cosmic noon - peak quasar era"
    if t < 9.0 * GYR:
        return "Galaxy assembly"
    if t < PRESENT_AGE * 1.02:
        return "Present day"

    return "Dark-energy dominated future"
