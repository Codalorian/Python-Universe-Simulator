"""
Supernovae: classification, energetics and light curves.

Light curves are powered by the radioactive decay chain

    Ni-56  ->  Co-56  ->  Fe-56

with half-lives of 6.08 and 77.2 days. That single chain is what makes
every supernova in the sky fade on the same characteristic timescale,
and it is what this module integrates.
"""

from dataclasses import dataclass

import numpy as np

from simulation_constants import (
    DAY,
    YEAR,
    SUN_MASS,
    SUN_LUMINOSITY,
    C,
    FOE,
    CORE_COLLAPSE_MIN_MASS,
    DIRECT_COLLAPSE_MIN_MASS,
    PAIR_INSTABILITY_MIN_MASS,
    PAIR_INSTABILITY_MAX_MASS,
)

# Decay constants, in seconds.
TAU_NI56 = 6.075 * DAY / np.log(2.0)
TAU_CO56 = 77.24 * DAY / np.log(2.0)

# Specific decay power at t = 0, in W per kg of Ni-56.
#
#   Ni-56: 1.75 MeV per decay, half-life  6.075 d  ->  3.9e6 W/kg
#   Co-56: 3.73 MeV per decay, half-life 77.24  d  ->  6.8e5 W/kg
EPSILON_NI = 3.9e6
EPSILON_CO = 6.8e5

# Supernova type codes.
SN_TYPE_IA = 0
SN_TYPE_II = 1
SN_TYPE_IBC = 2
SN_TYPE_PAIR_INSTABILITY = 3
SN_TYPE_HYPERNOVA = 4

SN_TYPE_NAMES = {
    SN_TYPE_IA: "Type Ia",
    SN_TYPE_II: "Type II",
    SN_TYPE_IBC: "Type Ib/c",
    SN_TYPE_PAIR_INSTABILITY: "Pair-instability",
    SN_TYPE_HYPERNOVA: "Hypernova / collapsar",
}

SN_TYPE_DESCRIPTIONS = {
    SN_TYPE_IA: (
        "A white dwarf pushed over the Chandrasekhar limit. Carbon "
        "detonates and the entire star is unbound: no remnant."
    ),
    SN_TYPE_II: (
        "An iron core exceeds the Chandrasekhar mass and collapses in "
        "under a second. The rebound and neutrino heating blow off the "
        "hydrogen envelope, leaving a neutron star."
    ),
    SN_TYPE_IBC: (
        "A core collapse in a star already stripped of its hydrogen "
        "(and, for Ic, helium) envelope by winds or a companion."
    ),
    SN_TYPE_PAIR_INSTABILITY: (
        "In a very massive core, photons convert to electron-positron "
        "pairs, pressure support fails, and oxygen burning detonates "
        "the entire star. Nothing is left behind."
    ),
    SN_TYPE_HYPERNOVA: (
        "An extremely energetic collapse forming a black hole with an "
        "accretion disc and relativistic jets."
    ),
}


def classify(mass_solar, metallicity=0.02, from_white_dwarf=False):
    """
    Supernova type from progenitor mass.
    """

    m = np.asarray(mass_solar, dtype=np.float64)

    if from_white_dwarf:
        return np.full(m.shape, SN_TYPE_IA, dtype=np.int8)

    # Metal-rich massive stars lose their hydrogen envelope to line-
    # driven winds and explode as stripped-envelope Ib/c events.
    z = np.asarray(metallicity, dtype=np.float64)
    stripped = (m >= 20.0) & (z > 0.008)

    return np.select(
        [
            (m >= PAIR_INSTABILITY_MIN_MASS) & (m <= PAIR_INSTABILITY_MAX_MASS),
            m >= DIRECT_COLLAPSE_MIN_MASS,
            stripped,
            m >= CORE_COLLAPSE_MIN_MASS,
        ],
        [
            SN_TYPE_PAIR_INSTABILITY,
            SN_TYPE_HYPERNOVA,
            SN_TYPE_IBC,
            SN_TYPE_II,
        ],
        default=SN_TYPE_II,
    ).astype(np.int8)


def nickel_mass(mass_solar, sn_type):
    """
    Mass of radioactive Ni-56 synthesised, in solar masses. This sets
    the brightness of the entire light curve.
    """

    m = np.asarray(mass_solar, dtype=np.float64)
    t = np.asarray(sn_type)

    return np.select(
        [
            t == SN_TYPE_IA,
            t == SN_TYPE_PAIR_INSTABILITY,
            t == SN_TYPE_HYPERNOVA,
        ],
        [
            np.full_like(m, 0.6),
            # PISN nickel yield grows steeply with helium core mass.
            np.clip(0.4 * (m - 130.0), 0.1, 50.0),
            np.full_like(m, 0.3),
        ],
        default=np.clip(0.02 + 0.005 * (m - 8.0), 0.01, 0.2),
    )


def kinetic_energy(mass_solar, sn_type):
    """
    Explosion kinetic energy in joules.
    """

    m = np.asarray(mass_solar, dtype=np.float64)
    t = np.asarray(sn_type)

    return np.select(
        [
            t == SN_TYPE_IA,
            t == SN_TYPE_PAIR_INSTABILITY,
            t == SN_TYPE_HYPERNOVA,
        ],
        [
            np.full_like(m, 1.5 * FOE),
            np.clip(0.5 * (m - 100.0), 1.0, 100.0) * FOE,
            np.full_like(m, 30.0 * FOE),
        ],
        default=np.full_like(m, 1.0 * FOE),
    )


def neutrino_energy(sn_type):
    """
    Core collapse radiates ~3e46 J in neutrinos, a hundred times the
    kinetic energy and a hundred thousand times the light. Thermonuclear
    events (Ia, pair-instability) never form a proto-neutron star and
    release almost none.
    """

    t = np.asarray(sn_type)

    core_collapse = (t == SN_TYPE_II) | (t == SN_TYPE_IBC) | (t == SN_TYPE_HYPERNOVA)

    return np.where(core_collapse, 3.0e46, 0.0)


def radioactive_luminosity(time_since_explosion, nickel_mass_solar):
    """
    Bolometric luminosity from the Ni-56 -> Co-56 -> Fe-56 chain, in
    watts. Vectorised over any number of simultaneous supernovae.
    """

    t = np.maximum(np.asarray(time_since_explosion, dtype=np.float64), 0.0)
    m_ni = np.asarray(nickel_mass_solar, dtype=np.float64) * SUN_MASS

    decay_ni = np.exp(-t / TAU_NI56)
    decay_co = (TAU_CO56 / (TAU_CO56 - TAU_NI56)) * (
        np.exp(-t / TAU_CO56) - np.exp(-t / TAU_NI56)
    )

    return m_ni * (EPSILON_NI * decay_ni + EPSILON_CO * decay_co)


def light_curve(time_since_explosion, nickel_mass_solar, sn_type, rise_time=None):
    """
    Bolometric luminosity in solar luminosities, including the finite
    rise while the ejecta are still optically thick.

    Photons diffuse out of the expanding ejecta over a few weeks, so
    the observed curve is the radioactive input smoothed by a rising
    escape fraction.
    """

    t = np.maximum(np.asarray(time_since_explosion, dtype=np.float64), 0.0)
    ty = np.asarray(sn_type)

    if rise_time is None:
        rise_time = np.where(
            ty == SN_TYPE_PAIR_INSTABILITY,
            120.0 * DAY,
            np.where(ty == SN_TYPE_IA, 18.0 * DAY, 25.0 * DAY),
        )

    escape = 1.0 - np.exp(-((t / rise_time) ** 2))

    watts = radioactive_luminosity(t, nickel_mass_solar) * escape

    # Type II supernovae also have a hydrogen recombination plateau:
    # ~100 days at roughly 1e8 Lsun, powered by stored shock energy.
    plateau = np.where(
        ty == SN_TYPE_II,
        1.0e8 * SUN_LUMINOSITY * np.exp(-((t / (100.0 * DAY)) ** 4)),
        0.0,
    )

    # Shock breakout: hours of extreme ultraviolet as the shock
    # reaches the surface. Thermonuclear events have no shock to break
    # out, so this applies to core collapse only.
    is_core_collapse = (ty == SN_TYPE_II) | (ty == SN_TYPE_IBC) | (ty == SN_TYPE_HYPERNOVA)

    breakout = np.where(
        is_core_collapse,
        1.0e11 * SUN_LUMINOSITY * np.exp(-t / (2.0 * 3600.0)),
        0.0,
    )

    return (watts + plateau + breakout) / SUN_LUMINOSITY


def peak_luminosity_solar(nickel_mass_solar, sn_type):
    """
    Arnett's rule: the peak luminosity equals the instantaneous decay
    power at the time of peak.
    """

    ty = np.asarray(sn_type)

    peak_time = np.where(
        ty == SN_TYPE_PAIR_INSTABILITY,
        120.0 * DAY,
        np.where(ty == SN_TYPE_IA, 18.0 * DAY, 25.0 * DAY),
    )

    return radioactive_luminosity(peak_time, nickel_mass_solar) / SUN_LUMINOSITY


def absolute_magnitude(luminosity_solar):
    """
    Bolometric absolute magnitude, M = 4.74 - 2.5 log10(L / Lsun).
    """

    lum = np.maximum(np.asarray(luminosity_solar, dtype=np.float64), 1.0e-30)

    return 4.74 - 2.5 * np.log10(lum)


def ejecta_velocity(mass_solar, sn_type):
    """
    Characteristic ejecta speed, from v = sqrt(2 E / M_ejecta).
    """

    m = np.asarray(mass_solar, dtype=np.float64)
    energy = kinetic_energy(m, sn_type)

    ejecta = np.maximum(m * 0.8, 0.1) * SUN_MASS

    return np.minimum(np.sqrt(2.0 * energy / ejecta), 0.3 * C)


def remnant_radius(time_since_explosion, mass_solar, sn_type, ambient_density=2.0e-21):
    """
    Blast-wave radius in metres.

    Free expansion at first, then the Sedov-Taylor similarity solution
    R = 1.17 (E t^2 / rho)^(1/5) once the swept-up mass dominates.
    """

    t = np.maximum(np.asarray(time_since_explosion, dtype=np.float64), 1.0)

    v = ejecta_velocity(mass_solar, sn_type)
    energy = kinetic_energy(mass_solar, sn_type)

    free = v * t
    sedov = 1.17 * (energy * t * t / ambient_density) ** 0.2

    return np.minimum(free, sedov)


@dataclass(slots=True)
class SupernovaEvent:
    """
    A single explosion, tracked by the renderer until it fades.
    """

    star_index: int
    sn_type: int
    explosion_time: float
    progenitor_mass: float
    nickel_mass: float
    kinetic_energy: float
    position: tuple
    manual: bool = False

    def luminosity_at(self, cosmic_time) -> float:
        dt = max(0.0, cosmic_time - self.explosion_time)

        return float(light_curve(dt, self.nickel_mass, self.sn_type))

    def is_visible(self, cosmic_time) -> bool:
        """
        A supernova stays naked-eye bright for a couple of years.
        """

        return (cosmic_time - self.explosion_time) < 5.0 * YEAR

    @property
    def type_name(self) -> str:
        return SN_TYPE_NAMES[int(self.sn_type)]

    @property
    def description(self) -> str:
        return SN_TYPE_DESCRIPTIONS[int(self.sn_type)]
