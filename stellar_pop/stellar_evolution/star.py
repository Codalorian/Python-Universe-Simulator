"""
A single star, as a plain object.

The simulation itself never uses this: it holds its population as a
struct of arrays in `population.py`, because a quarter of a million
Python objects cannot be evolved sixty times a second. This class is
for the cases where one star at a time is the right granularity -
inspecting something, testing a relation, or scripting against the
model from outside.

Every quantity here is delegated to `relations.py`, so there is one
definition of stellar structure in the codebase and this can never
drift away from what is on screen.
"""

from dataclasses import dataclass, field
from enum import Enum

from simulation_constants import (
    YEAR,
    GYR,
    SUN_RADIUS,
    CORE_COLLAPSE_MIN_MASS,
)

from stellar_pop.stellar_evolution import relations
from stellar_pop.stellar_evolution.blackbody import spectral_class, temperature_to_rgb

from stellar_pop.remnants import (
    remnant_mass,
    remnant_radius_solar,
    remnant_temperature,
)

from stellar_pop.supernovae import (
    classify as classify_supernova,
    nickel_mass,
    light_curve,
    SN_TYPE_NAMES,
)


class StellarState(Enum):
    NOT_YET_FORMED = "not_yet_formed"
    MAIN_SEQUENCE = "main_sequence"
    GIANT = "giant"
    WHITE_DWARF = "white_dwarf"
    NEUTRON_STAR = "neutron_star"
    BLACK_HOLE = "black_hole"
    NOTHING = "nothing"


_FATE_STATES = {
    relations.FATE_WHITE_DWARF: StellarState.WHITE_DWARF,
    relations.FATE_NEUTRON_STAR: StellarState.NEUTRON_STAR,
    relations.FATE_BLACK_HOLE: StellarState.BLACK_HOLE,
    relations.FATE_DIRECT_COLLAPSE: StellarState.BLACK_HOLE,
    relations.FATE_PAIR_INSTABILITY: StellarState.NOTHING,
}


@dataclass
class Star:
    """
    A star defined by the three things that actually determine its
    life: mass, composition, and when it was born.

    Everything else - luminosity, radius, temperature, how long it
    lives and what it leaves behind - follows, and is recomputed from
    cosmic time rather than accumulated, so asking about any epoch is
    exact and order-independent.
    """

    mass_solar: float
    formation_time: float
    metallicity: float = 0.0142

    identifier: int = 0

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    exploded_early_at: float = field(default=None, repr=False)

    # --------------------------------------------------------
    # Timescales
    # --------------------------------------------------------

    @property
    def main_sequence_lifetime(self) -> float:
        return float(
            relations.main_sequence_lifetime(self.mass_solar, self.metallicity)
        )

    @property
    def giant_duration(self) -> float:
        return float(
            relations.giant_phase_duration(
                self.mass_solar, self.main_sequence_lifetime
            )
        )

    @property
    def giant_start(self) -> float:
        return self.formation_time + self.main_sequence_lifetime

    @property
    def death_time(self) -> float:
        if self.exploded_early_at is not None:
            return self.exploded_early_at

        return self.giant_start + self.giant_duration

    @property
    def fate(self) -> int:
        return int(relations.stellar_fate(self.mass_solar))

    @property
    def fate_name(self) -> str:
        return relations.FATE_NAMES[self.fate]

    @property
    def can_core_collapse(self) -> bool:
        return self.mass_solar >= CORE_COLLAPSE_MIN_MASS

    # --------------------------------------------------------
    # State at a given cosmic time
    # --------------------------------------------------------

    def age_at(self, cosmic_time: float) -> float:
        return max(0.0, cosmic_time - self.formation_time)

    def state_at(self, cosmic_time: float) -> StellarState:
        if cosmic_time < self.formation_time:
            return StellarState.NOT_YET_FORMED

        # A detonated star is dead from that moment on, whatever its
        # natural timeline said it should still be doing.
        if cosmic_time >= self.death_time:
            return _FATE_STATES[self.fate]

        if cosmic_time < self.giant_start:
            return StellarState.MAIN_SEQUENCE

        return StellarState.GIANT

    def luminosity_at(self, cosmic_time: float) -> float:
        """
        In solar luminosities, including any supernova still fading.
        """

        state = self.state_at(cosmic_time)

        if state is StellarState.NOT_YET_FORMED:
            return 0.0

        if state is StellarState.MAIN_SEQUENCE:
            return float(relations.main_sequence_luminosity(self.mass_solar))

        if state is StellarState.GIANT:
            return float(
                relations.giant_luminosity(
                    self.mass_solar, self._giant_fraction(cosmic_time)
                )
            )

        radius = self.radius_at(cosmic_time)
        temperature = self.temperature_at(cosmic_time)

        from stellar_pop.remnants import remnant_luminosity_solar

        quiescent = float(remnant_luminosity_solar(radius, temperature))

        return quiescent + self.supernova_luminosity_at(cosmic_time)

    def radius_at(self, cosmic_time: float) -> float:
        """
        In solar radii. For a black hole this is the event horizon.
        """

        state = self.state_at(cosmic_time)

        if state is StellarState.NOT_YET_FORMED:
            return 0.0

        if state is StellarState.MAIN_SEQUENCE:
            return float(relations.main_sequence_radius(self.mass_solar))

        if state is StellarState.GIANT:
            return float(
                relations.giant_radius(
                    self.mass_solar, self._giant_fraction(cosmic_time)
                )
            )

        return float(remnant_radius_solar(self.remnant_mass, self.fate))

    def temperature_at(self, cosmic_time: float) -> float:
        state = self.state_at(cosmic_time)

        if state is StellarState.NOT_YET_FORMED:
            return 0.0

        if state in (StellarState.MAIN_SEQUENCE, StellarState.GIANT):
            return float(
                relations.effective_temperature(
                    self.luminosity_at(cosmic_time), self.radius_at(cosmic_time)
                )
            )

        return float(
            remnant_temperature(
                cosmic_time - self.death_time, self.remnant_mass, self.fate
            )
        )

    def colour_at(self, cosmic_time: float):
        """
        sRGB, from the true CIE colour of the star's Planck spectrum.
        """

        return temperature_to_rgb(max(self.temperature_at(cosmic_time), 500.0))

    def spectral_class_at(self, cosmic_time: float) -> str:
        return spectral_class(self.temperature_at(cosmic_time))

    def _giant_fraction(self, cosmic_time: float) -> float:
        duration = self.giant_duration

        if duration <= 0.0:
            return 1.0

        return min(max((cosmic_time - self.giant_start) / duration, 0.0), 1.0)

    # --------------------------------------------------------
    # Remnant and supernova
    # --------------------------------------------------------

    @property
    def remnant_mass(self) -> float:
        return float(remnant_mass(self.mass_solar, self.fate))

    @property
    def supernova_type(self):
        if not self.explodes:
            return None

        return int(classify_supernova(self.mass_solar, self.metallicity))

    @property
    def supernova_type_name(self):
        code = self.supernova_type

        return None if code is None else SN_TYPE_NAMES[code]

    @property
    def explodes(self) -> bool:
        return self.fate in (
            relations.FATE_NEUTRON_STAR,
            relations.FATE_BLACK_HOLE,
            relations.FATE_PAIR_INSTABILITY,
        )

    def supernova_luminosity_at(self, cosmic_time: float) -> float:
        """
        Light curve of the explosion, in solar luminosities. Zero for a
        star that never explodes or has not yet died.
        """

        if not self.explodes or cosmic_time < self.death_time:
            return 0.0

        sn_type = classify_supernova(self.mass_solar, self.metallicity)

        return float(
            light_curve(
                cosmic_time - self.death_time,
                nickel_mass(self.mass_solar, sn_type),
                sn_type,
            )
        )

    def force_supernova(self, cosmic_time: float):
        """
        Detonate the star now, if it is massive enough and still
        burning. Returns the supernova type, or None.
        """

        if not self.can_core_collapse:
            return None

        if self.state_at(cosmic_time) not in (
            StellarState.MAIN_SEQUENCE,
            StellarState.GIANT,
        ):
            return None

        self.exploded_early_at = cosmic_time

        return self.supernova_type_name

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    def describe(self, cosmic_time: float) -> str:
        return (
            f"Star {self.identifier}: {self.mass_solar:.3f} Msun "
            f"{self.spectral_class_at(cosmic_time)} "
            f"{self.state_at(cosmic_time).value}, "
            f"L={self.luminosity_at(cosmic_time):.4g} Lsun, "
            f"R={self.radius_at(cosmic_time):.4g} Rsun, "
            f"T={self.temperature_at(cosmic_time):,.0f} K, "
            f"born {self.formation_time / GYR:.3f} Gyr, "
            f"ends as a {self.fate_name}"
        )
