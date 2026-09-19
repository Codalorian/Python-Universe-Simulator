"""
The early universe: the thermal history from the Planck epoch to the
end of reionisation.

Temperatures come from the radiation-dominated relation

    kT ~ 1.5 MeV * (t / 1 s)^-1/2

for t below matter-radiation equality, and from the CMB scaling
T = 2.7255 K / a(t) afterwards. The two agree to within a factor of a
few at the crossover, which is as much as a single-line formula can
promise.
"""

from dataclasses import dataclass

import numpy as np

from simulation_constants import (
    K_B,
    YEAR,
    MYR,
    GYR,
    T_PLANCK_EPOCH,
    T_INFLATION_END,
    T_QUARK_HADRON,
    T_NUCLEOSYNTHESIS,
    T_MATTER_RADIATION_EQ,
    T_RECOMBINATION,
    T_FIRST_STARS,
    T_REIONIZATION_END,
)

EV = 1.602176634e-19
MEV = 1.0e6 * EV


@dataclass(frozen=True)
class Epoch:
    name: str
    start_time: float
    description: str


# Ordered by start time.
EPOCHS = (
    Epoch(
        "Planck epoch",
        0.0,
        "Gravity is not separable from the other forces. No known "
        "physics applies.",
    ),
    Epoch(
        "Grand unification",
        T_PLANCK_EPOCH,
        "Strong, weak and electromagnetic forces are one interaction.",
    ),
    Epoch(
        "Inflation",
        1.0e-36,
        "Space expands by a factor of ~1e26 in ~1e-32 s, flattening "
        "the universe and seeding structure from quantum fluctuations.",
    ),
    Epoch(
        "Electroweak epoch",
        T_INFLATION_END,
        "Quark-gluon plasma. The Higgs field has not yet condensed.",
    ),
    Epoch(
        "Quark epoch",
        1.0e-12,
        "Electroweak symmetry breaks; particles acquire mass.",
    ),
    Epoch(
        "Hadron epoch",
        T_QUARK_HADRON,
        "Quarks bind into protons and neutrons. Matter narrowly "
        "outnumbers antimatter and the rest annihilates.",
    ),
    Epoch(
        "Lepton epoch",
        1.0,
        "Neutrinos decouple and free-stream. Electron-positron pairs "
        "annihilate, reheating the photons.",
    ),
    Epoch(
        "Nucleosynthesis",
        T_NUCLEOSYNTHESIS,
        "Protons and neutrons fuse into helium-4, deuterium and "
        "lithium-7. Roughly 24% of baryonic mass becomes helium.",
    ),
    Epoch(
        "Photon epoch",
        1200.0,
        "A hot opaque plasma of nuclei, electrons and photons.",
    ),
    Epoch(
        "Matter-radiation equality",
        T_MATTER_RADIATION_EQ,
        "Matter density overtakes radiation density; density "
        "perturbations can finally grow.",
    ),
    Epoch(
        "Recombination",
        T_RECOMBINATION,
        "Electrons bind to nuclei, the universe turns transparent and "
        "the cosmic microwave background is released.",
    ),
    Epoch(
        "Dark Ages",
        400_000.0 * YEAR,
        "No stars yet. Only the fading CMB and neutral hydrogen.",
    ),
    Epoch(
        "Cosmic dawn",
        T_FIRST_STARS,
        "Population III stars ignite in the first minihalos: metal-free, "
        "very massive, and very short-lived.",
    ),
    Epoch(
        "Reionisation",
        400.0 * MYR,
        "Ultraviolet light from the first stars and quasars re-ionises "
        "the intergalactic medium.",
    ),
    Epoch(
        "Galaxy assembly",
        T_REIONIZATION_END,
        "Hierarchical merging builds the galaxies we see today.",
    ),
    Epoch(
        "Cosmic noon",
        2.0 * GYR,
        "Peak star formation and peak quasar activity.",
    ),
    Epoch(
        "Present day",
        10.0 * GYR,
        "Star formation has declined by an order of magnitude. Dark "
        "energy now dominates the energy budget.",
    ),
)


def current_epoch(cosmic_time: float) -> Epoch:
    """
    The last epoch whose start time has passed.
    """

    result = EPOCHS[0]

    for epoch in EPOCHS:
        if cosmic_time >= epoch.start_time:
            result = epoch
        else:
            break

    return result


def radiation_temperature(cosmic_time):
    """
    Photon temperature in kelvin.

    Radiation-dominated scaling below equality, CMB scaling above it.
    """

    t = np.maximum(np.asarray(cosmic_time, dtype=np.float64), 1.0e-44)

    # kT ~ 1.5 MeV (t/s)^-1/2 during radiation domination.
    early = 1.5 * MEV / np.sqrt(t) / K_B

    # a ~ t^1/2 in radiation domination, normalised at equality so the
    # two branches join continuously.
    t_eq = T_MATTER_RADIATION_EQ
    late_at_eq = 1.5 * MEV / np.sqrt(t_eq) / K_B

    # Beyond equality matter dominates and a ~ t^2/3, so T ~ t^-2/3.
    late = late_at_eq * (t / t_eq) ** (-2.0 / 3.0)

    return np.where(t < t_eq, early, late)


def photon_energy_ev(cosmic_time):
    return K_B * radiation_temperature(cosmic_time) / EV


def is_transparent(cosmic_time) -> bool:
    """
    The universe is opaque to photons before recombination.
    """

    return cosmic_time >= T_RECOMBINATION


def helium_mass_fraction(cosmic_time):
    """
    Primordial helium builds up over the few minutes of Big Bang
    nucleosynthesis and then stays fixed at Y ~ 0.247.
    """

    t = np.asarray(cosmic_time, dtype=np.float64)

    ramp = np.clip((t - T_NUCLEOSYNTHESIS) / (1200.0 - T_NUCLEOSYNTHESIS), 0.0, 1.0)

    return 0.247 * ramp


def ionised_fraction(cosmic_time):
    """
    Fraction of hydrogen that is ionised.

    Fully ionised before recombination, essentially neutral through the
    Dark Ages, then driven back to ~1 by the first stars and quasars.
    """

    t = np.asarray(cosmic_time, dtype=np.float64)

    reionised = np.clip(
        (t - T_FIRST_STARS) / (T_REIONIZATION_END - T_FIRST_STARS), 0.0, 1.0
    )

    return np.where(t < T_RECOMBINATION, 1.0, reionised)


def cmb_glow_intensity(cosmic_time) -> float:
    """
    A 0-1 value for how strongly the renderer should paint the sky with
    the primordial fireball. It peaks at recombination, when the CMB is
    actually released at ~3000 K, and fades as the universe expands.
    """

    t = float(cosmic_time)

    if t <= 0.0:
        return 1.0

    if t < T_RECOMBINATION:
        # Opaque plasma: brilliant everywhere.
        return 1.0

    # Fade out over the following few hundred million years.
    fade = (t - T_RECOMBINATION) / (300.0 * MYR)

    return float(max(0.0, 1.0 - fade))
