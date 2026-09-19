"""
Planet formation by core accretion.

A protoplanetary disc is laid down with a minimum-mass-solar-nebula
surface density, the snow line is placed where water ice condenses,
and planetesimals grow until they either stay rocky or run away into
gas giants. The resulting architectures reproduce the observed
pattern: rocky worlds inside the snow line, giants just outside it,
ice giants further out.

Systems are generated deterministically from a star's index, so the
same star always has the same planets no matter when you fly past it.
"""

from dataclasses import dataclass, field

import numpy as np

from simulation_constants import (
    AU,
    G,
    SUN_MASS,
    EARTH_MASS,
    EARTH_RADIUS,
    JUPITER_MASS,
)

from planetary_systems.orbital_mechanics import (
    orbital_period,
    position_at_time,
)

# Planet type codes.
PLANET_ROCKY = 0
PLANET_OCEAN = 1
PLANET_GAS_GIANT = 2
PLANET_ICE_GIANT = 3
PLANET_HOT_JUPITER = 4
PLANET_LAVA = 5
PLANET_ICE = 6

PLANET_TYPE_NAMES = {
    PLANET_ROCKY: "rocky",
    PLANET_OCEAN: "ocean world",
    PLANET_GAS_GIANT: "gas giant",
    PLANET_ICE_GIANT: "ice giant",
    PLANET_HOT_JUPITER: "hot Jupiter",
    PLANET_LAVA: "lava world",
    PLANET_ICE: "ice world",
}

# Approximate surface colours, for the renderer.
PLANET_COLOURS = {
    PLANET_ROCKY: (0.62, 0.55, 0.47),
    PLANET_OCEAN: (0.22, 0.42, 0.72),
    PLANET_GAS_GIANT: (0.80, 0.68, 0.50),
    PLANET_ICE_GIANT: (0.42, 0.66, 0.80),
    PLANET_HOT_JUPITER: (0.85, 0.45, 0.28),
    PLANET_LAVA: (0.72, 0.24, 0.12),
    PLANET_ICE: (0.82, 0.86, 0.92),
}


def snow_line_au(luminosity_solar):
    """
    Distance at which the disc is cold enough for water to freeze,
    around 170 K. Scales as sqrt(L), so it sits at 2.7 AU for the Sun.
    """

    return 2.7 * np.sqrt(np.maximum(np.asarray(luminosity_solar, dtype=np.float64), 1e-9))


def disc_surface_density(radius_au, disc_mass_solar=0.01):
    """
    Minimum-mass solar nebula profile, Sigma ~ r^-3/2, in kg/m^2.
    """

    r = np.maximum(np.asarray(radius_au, dtype=np.float64), 0.05)

    return 1700.0 * (disc_mass_solar / 0.01) * r**-1.5


def equilibrium_temperature(luminosity_solar, distance_au, albedo=0.3):
    """
    Blackbody equilibrium temperature of a planet, in kelvin.

        T = 278 K * (1-A)^0.25 * L^0.25 / sqrt(d_AU)

    Gives 255 K for Earth, which is right: the extra 33 K is
    greenhouse warming.
    """

    lum = np.maximum(np.asarray(luminosity_solar, dtype=np.float64), 1e-12)
    d = np.maximum(np.asarray(distance_au, dtype=np.float64), 1.0e-4)

    return 278.6 * (1.0 - albedo) ** 0.25 * lum**0.25 / np.sqrt(d)


def planet_radius_earth(mass_earth, planet_type):
    """
    Mass-radius relation, in Earth radii.

    Rocky planets follow R ~ M^0.27; above ~10 Earth masses a body
    captures hydrogen and its radius jumps, then becomes almost
    independent of mass as electron degeneracy sets in, which is why
    Jupiter and a 10-Jupiter-mass planet are the same size.
    """

    m = np.maximum(np.asarray(mass_earth, dtype=np.float64), 0.01)
    t = np.asarray(planet_type)

    rocky = m**0.27
    giant = 11.2 * (m / 318.0) ** 0.06

    ice_giant = 3.9 * (m / 15.0) ** 0.30

    return np.select(
        [
            (t == PLANET_GAS_GIANT) | (t == PLANET_HOT_JUPITER),
            t == PLANET_ICE_GIANT,
        ],
        [giant, ice_giant],
        default=rocky,
    )


def classify_planet(mass_earth, distance_au, snow_line, equilibrium_t):
    """
    Planet type from mass, position and temperature.
    """

    m = np.asarray(mass_earth, dtype=np.float64)
    d = np.asarray(distance_au, dtype=np.float64)
    t = np.asarray(equilibrium_t, dtype=np.float64)

    return np.select(
        [
            (m > 50.0) & (d < 0.1),
            m > 50.0,
            m > 8.0,
            t > 1000.0,
            t < 180.0,
            (t >= 250.0) & (t <= 330.0) & (m > 0.4) & (m < 6.0),
        ],
        [
            PLANET_HOT_JUPITER,
            PLANET_GAS_GIANT,
            PLANET_ICE_GIANT,
            PLANET_LAVA,
            PLANET_ICE,
            PLANET_OCEAN,
        ],
        default=PLANET_ROCKY,
    ).astype(np.int8)


def isolation_mass_earth(radius_au, disc_mass_solar, star_mass_solar):
    """
    Mass a protoplanet reaches once it has swept its whole feeding
    zone, in Earth masses. This is the core-accretion bottleneck: only
    beyond the snow line, where solids are three times more abundant,
    does it exceed the ~10 Earth masses needed to capture gas.
    """

    r = np.maximum(np.asarray(radius_au, dtype=np.float64), 0.05)

    sigma = disc_surface_density(r, disc_mass_solar)

    # Feeding zone width ~ a few Hill radii.
    a_m = r * AU
    m_star = np.asarray(star_mass_solar, dtype=np.float64) * SUN_MASS

    mass = (8.0 * np.pi * sigma * a_m * a_m) ** 1.5 / np.sqrt(3.0 * m_star) / EARTH_MASS

    return mass


@dataclass(slots=True)
class Planet:
    name: str
    mass_earth: float
    radius_earth: float
    semi_major_axis_au: float
    eccentricity: float
    inclination: float
    longitude_of_node: float
    argument_of_periapsis: float
    mean_anomaly_epoch: float
    planet_type: int
    equilibrium_temperature: float
    period_seconds: float
    moons: int = 0

    @property
    def type_name(self) -> str:
        return PLANET_TYPE_NAMES[int(self.planet_type)]

    @property
    def colour(self):
        return PLANET_COLOURS[int(self.planet_type)]

    @property
    def mass_jupiter(self) -> float:
        return self.mass_earth * EARTH_MASS / JUPITER_MASS

    @property
    def surface_gravity(self) -> float:
        return (
            G
            * self.mass_earth
            * EARTH_MASS
            / (self.radius_earth * EARTH_RADIUS) ** 2
        )

    @property
    def is_habitable(self) -> bool:
        return (
            self.planet_type in (PLANET_ROCKY, PLANET_OCEAN)
            and 250.0 <= self.equilibrium_temperature <= 330.0
            and 0.3 <= self.mass_earth <= 8.0
        )


@dataclass(slots=True)
class PlanetarySystem:
    star_index: int
    star_mass_solar: float
    star_luminosity_solar: float
    planets: list = field(default_factory=list)

    def positions_at(self, time_seconds):
        """
        Heliocentric positions of every planet, in metres, as (N, 3).
        """

        if not self.planets:
            return np.zeros((0, 3))

        a = np.array([p.semi_major_axis_au for p in self.planets]) * AU
        e = np.array([p.eccentricity for p in self.planets])
        inc = np.array([p.inclination for p in self.planets])
        node = np.array([p.longitude_of_node for p in self.planets])
        peri = np.array([p.argument_of_periapsis for p in self.planets])
        m0 = np.array([p.mean_anomaly_epoch for p in self.planets])

        return position_at_time(
            time_seconds, a, e, inc, node, peri, m0, self.star_mass_solar * SUN_MASS
        )

    @property
    def habitable_planets(self):
        return [p for p in self.planets if p.is_habitable]


def generate_system(star_index, mass_solar, luminosity_solar, metallicity, seed=0):
    """
    Build a planetary system for one star, deterministically.

    The giant planet occurrence rate rises steeply with host
    metallicity, an observed correlation that falls straight out of
    core accretion: more metals means more solids means cores that
    form before the gas disc disperses.
    """

    rng = np.random.default_rng((int(star_index) * 2_654_435_761 + int(seed)) % (2**63))

    # Massive stars have short-lived discs; the very hottest destroy
    # them before planets can form.
    if mass_solar > 20.0:
        return PlanetarySystem(star_index, mass_solar, luminosity_solar, [])

    metal_factor = float(np.clip(metallicity / 0.0142, 0.05, 4.0))

    if rng.random() > min(0.25 + 0.55 * metal_factor, 0.95):
        return PlanetarySystem(star_index, mass_solar, luminosity_solar, [])

    n_planets = int(rng.integers(1, 9))

    disc_mass = 0.01 * mass_solar * (0.4 + 1.2 * rng.random()) * metal_factor**0.5

    snow = float(snow_line_au(luminosity_solar))

    # Orbits are laid out in an approximately geometric progression,
    # which is what dynamical packing produces.
    inner = 0.03 + 0.25 * rng.random() * np.sqrt(max(luminosity_solar, 0.01))
    spacing = 1.4 + 0.9 * rng.random(n_planets)

    axes = inner * np.cumprod(spacing)

    planets = []

    for index in range(n_planets):
        a_au = float(axes[index])

        if a_au > 200.0:
            break

        iso = float(isolation_mass_earth(a_au, disc_mass, mass_solar))

        # Beyond the snow line, ices triple the solid surface density,
        # so cores grow far larger and can capture hydrogen.
        beyond_snow = a_au > snow

        mass = iso * (3.0 if beyond_snow else 1.0)

        mass *= float(np.exp(rng.normal(0.0, 0.6)))

        if beyond_snow and mass > 10.0 and rng.random() < 0.6:
            # Runaway gas accretion.
            mass *= float(10.0 ** rng.uniform(0.6, 1.6))

        mass = float(np.clip(mass, 0.02, 4000.0))

        eccentricity = float(np.clip(abs(rng.normal(0.0, 0.08)), 0.0, 0.7))

        t_eq = float(equilibrium_temperature(luminosity_solar, a_au))

        ptype = int(classify_planet(mass, a_au, snow, t_eq))

        radius = float(planet_radius_earth(mass, ptype))

        period = float(orbital_period(a_au * AU, mass_solar * SUN_MASS))

        # Big planets far from their star keep large moon systems.
        moons = int(rng.integers(0, 4)) if mass < 8 else int(rng.integers(2, 20))

        planets.append(
            Planet(
                name=f"{_roman(index + 1)}",
                mass_earth=mass,
                radius_earth=radius,
                semi_major_axis_au=a_au,
                eccentricity=eccentricity,
                inclination=float(rng.normal(0.0, 0.03)),
                longitude_of_node=float(rng.random() * 2.0 * np.pi),
                argument_of_periapsis=float(rng.random() * 2.0 * np.pi),
                mean_anomaly_epoch=float(rng.random() * 2.0 * np.pi),
                planet_type=ptype,
                equilibrium_temperature=t_eq,
                period_seconds=period,
                moons=moons,
            )
        )

    return PlanetarySystem(star_index, mass_solar, luminosity_solar, planets)


_ROMAN = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
)


def _roman(n: int) -> str:
    return _ROMAN[n - 1] if 1 <= n <= len(_ROMAN) else str(n)
