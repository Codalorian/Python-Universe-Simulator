"""
Galaxy formation: turning a catalogue of dark matter haloes into
galaxies with morphologies, sizes, colours and central black holes,
and laying their stars out in space.

Everything is generated as whole arrays. Building sixty thousand
galaxies is a few dozen NumPy calls, not sixty thousand Python loops.
"""

import numpy as np

from simulation_constants import (
    KPC,
    SUN_MASS,
)

from galaxy_pop.dark_matter_halos import (
    sample_halo_masses,
    virial_radius,
    virial_velocity,
    velocity_dispersion,
    concentration,
    stellar_mass_from_halo,
)

from compact_objects.black_holes import smbh_mass_from_bulge

# Morphology codes.
MORPH_SPIRAL = 0
MORPH_ELLIPTICAL = 1
MORPH_IRREGULAR = 2
MORPH_LENTICULAR = 3
MORPH_DWARF = 4

MORPH_NAMES = {
    MORPH_SPIRAL: "spiral",
    MORPH_ELLIPTICAL: "elliptical",
    MORPH_IRREGULAR: "irregular",
    MORPH_LENTICULAR: "lenticular",
    MORPH_DWARF: "dwarf spheroidal",
}


def classify_morphology(halo_mass_solar, rng):
    """
    Morphology from halo mass, following the observed morphology-mass
    relation: dwarfs below ~1e11, spirals through the middle, and
    ellipticals dominating the group and cluster scale where mergers
    have destroyed every disc.
    """

    m = np.asarray(halo_mass_solar, dtype=np.float64)
    u = rng.random(m.shape)

    # Probability of being an early-type (elliptical/lenticular), which
    # rises steeply with mass.
    p_early = np.clip(0.08 + 0.55 * np.log10(np.maximum(m, 1e8) / 1.0e11) / 3.0, 0.02, 0.85)

    morph = np.where(
        m < 3.0e10,
        MORPH_DWARF,
        np.where(
            u < p_early,
            np.where(rng.random(m.shape) < 0.7, MORPH_ELLIPTICAL, MORPH_LENTICULAR),
            np.where(rng.random(m.shape) < 0.85, MORPH_SPIRAL, MORPH_IRREGULAR),
        ),
    )

    return morph.astype(np.int8)


def disc_scale_length(stellar_mass_solar):
    """
    Exponential disc scale length in metres.

    Observed size-mass relation for late types: R_d ~ 3.5 kpc at the
    mass of the Milky Way, growing roughly as M*^0.3.
    """

    m = np.maximum(np.asarray(stellar_mass_solar, dtype=np.float64), 1.0e6)

    return 3.5 * KPC * (m / 5.0e10) ** 0.30


def effective_radius(stellar_mass_solar, morphology):
    """
    Half-light radius in metres. Ellipticals of a given stellar mass
    are more compact than discs at low mass and far larger at high
    mass, following the observed size-mass relations.
    """

    m = np.maximum(np.asarray(stellar_mass_solar, dtype=np.float64), 1.0e6)
    morph = np.asarray(morphology)

    late = 1.7 * disc_scale_length(m)
    early = 2.5 * KPC * (m / 1.0e10) ** 0.56

    return np.where((morph == MORPH_ELLIPTICAL) | (morph == MORPH_LENTICULAR), early, late)


def formation_redshift(halo_mass_solar, rng):
    """
    Redshift at which the halo assembled half its present mass.

    Small haloes collapse first. This is hierarchical structure
    formation: the universe builds from the bottom up.
    """

    m = np.maximum(np.asarray(halo_mass_solar, dtype=np.float64), 1.0e8)

    z_mean = 4.0 - 0.9 * np.log10(m / 1.0e11)

    return np.clip(z_mean + rng.normal(0.0, 0.7, m.shape), 0.2, 20.0)


def sample_positions(count, rng, radius_mpc, clustering=0.55):
    """
    Comoving positions in Mpc, as an (N, 3) float32 array.

    Galaxies are not scattered uniformly: they trace a cosmic web of
    filaments, sheets and voids. This is reproduced by drawing a modest
    number of filament seeds and placing most galaxies near the line
    segments between them, which produces the right visual texture
    without running an N-body code.
    """

    n_nodes = max(24, int(count**0.36))

    nodes = rng.normal(0.0, radius_mpc * 0.42, size=(n_nodes, 3))

    # Each galaxy is assigned to a filament joining two nearby nodes.
    a_idx = rng.integers(0, n_nodes, count)

    # Pick the second endpoint from among the nearest few nodes so the
    # filaments stay short and the web stays connected.
    offsets = rng.integers(1, 5, count)
    b_idx = (a_idx + offsets) % n_nodes

    t = rng.random((count, 1))

    spine = nodes[a_idx] * (1.0 - t) + nodes[b_idx] * t

    # Scatter perpendicular to the filament: tight for most galaxies,
    # with a broad tail filling the voids.
    tight = rng.normal(0.0, radius_mpc * 0.012, size=(count, 3))
    loose = rng.normal(0.0, radius_mpc * 0.10, size=(count, 3))

    in_filament = (rng.random((count, 1)) < clustering)

    positions = spine + np.where(in_filament, tight, loose)

    # Trim anything that wandered outside the simulated sphere.
    r = np.linalg.norm(positions, axis=1, keepdims=True)
    too_far = r > radius_mpc

    positions = np.where(too_far, positions * (radius_mpc / np.maximum(r, 1e-9)) * 0.98, positions)

    return positions.astype(np.float32)


def generate_catalogue(count, rng, expansion, radius_mpc, min_halo_mass=1.0e11):
    """
    Build the full galaxy catalogue.

    Returns a dict of parallel arrays: one entry per galaxy, all
    NumPy, all the same length.

    min_halo_mass acts as the survey selection limit. By number the
    real universe is overwhelmingly dwarf galaxies, but they are far
    too faint to see at cosmological distances, so a catalogue drawn
    all the way down to 1e9 Msun would be 95% objects that should not
    be visible. Cutting at 1e11 Msun reproduces what a deep flux-
    limited survey actually contains.
    """

    halo_mass = sample_halo_masses(count, rng, m_min=min_halo_mass)

    morphology = classify_morphology(halo_mass, rng)

    stellar_mass = stellar_mass_from_halo(halo_mass)

    # Ellipticals sit in haloes that converted a little more of their
    # gas before quenching; irregulars rather less.
    stellar_mass = stellar_mass * np.where(
        morphology == MORPH_ELLIPTICAL,
        1.3,
        np.where(morphology == MORPH_IRREGULAR, 0.7, 1.0),
    )

    z_form = formation_redshift(halo_mass, rng)
    t_form = expansion.time_at_redshift(z_form)

    radius = effective_radius(stellar_mass, morphology)

    positions = sample_positions(count, rng, radius_mpc)

    # Bulge fraction drives the central black hole mass.
    bulge_fraction = np.select(
        [
            morphology == MORPH_ELLIPTICAL,
            morphology == MORPH_LENTICULAR,
            morphology == MORPH_SPIRAL,
            morphology == MORPH_IRREGULAR,
        ],
        [1.0, 0.6, 0.2, 0.05],
        default=0.1,
    )

    smbh_mass = smbh_mass_from_bulge(stellar_mass * bulge_fraction)

    r_vir = virial_radius(halo_mass, expansion, expansion.present_age)
    v_vir = virial_velocity(halo_mass, r_vir)

    # Random orientation: a unit normal for the disc plane.
    orientation = rng.normal(size=(count, 3))
    orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)

    catalogue = {
        "halo_mass": halo_mass,
        "stellar_mass": stellar_mass,
        "morphology": morphology,
        "formation_redshift": z_form,
        "formation_time": t_form,
        "radius": radius,
        "position_mpc": positions,
        "smbh_mass": smbh_mass,
        "bulge_fraction": bulge_fraction,
        "virial_radius": r_vir,
        "virial_velocity": v_vir,
        "velocity_dispersion": velocity_dispersion(v_vir),
        "orientation": orientation.astype(np.float32),
        "spin_seed": rng.random(count).astype(np.float32),
    }

    return spatially_sort(catalogue)


def _spread_bits(value):
    """
    Interleave a 10-bit integer with two zero bits between each bit, so
    three of these can be OR-ed into a Morton code.
    """

    n = value.astype(np.int64) & 0x3FF

    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249

    return n


def morton_code(positions, resolution=1024):
    """
    Z-order curve index for each position.

    Points with nearby Morton codes are nearby in space, so sorting by
    this key turns a 3D neighbour search into a short scan along one
    array.
    """

    lo = positions.min(axis=0)
    hi = positions.max(axis=0)

    span = np.maximum(hi - lo, 1.0e-6)

    grid = np.clip(((positions - lo) / span * (resolution - 1)), 0, resolution - 1)
    grid = grid.astype(np.int64)

    return (
        _spread_bits(grid[:, 0])
        | (_spread_bits(grid[:, 1]) << 1)
        | (_spread_bits(grid[:, 2]) << 2)
    )


def spatially_sort(catalogue):
    """
    Reorder the catalogue along a Morton curve so that index-adjacent
    galaxies are also spatially adjacent.

    Neighbour queries then become a short index window instead of a
    search, and every per-galaxy pass gets better cache locality.
    """

    order = np.argsort(morton_code(catalogue["position_mpc"]), kind="stable")

    return {name: value[order] for name, value in catalogue.items()}


# ------------------------------------------------------------
# Laying out the stars of a single galaxy
# ------------------------------------------------------------

def spiral_disc_positions(count, rng, scale_length_m, arm_count=2, pitch=0.22, thickness=0.06):
    """
    Star positions for a spiral galaxy, in metres, as (N, 3).

    Radii follow an exponential disc; azimuths are drawn towards
    logarithmic spiral arms r = r0 exp(theta tan(pitch)), with a
    scatter that widens in the outer disc where the arms dissolve.
    """

    # Exponential surface density: invert the CDF of r exp(-r/h).
    u = rng.random(count)
    r = scale_length_m * _invert_exponential_disc(u)

    # Which arm, and where along it.
    arm = rng.integers(0, arm_count, count)

    theta_arm = np.log(np.maximum(r / (0.35 * scale_length_m), 1.0e-3)) / np.tan(pitch)

    theta = theta_arm + arm * (2.0 * np.pi / arm_count)

    # Arms are tight near the centre and wash out in the outskirts.
    spread = 0.25 + 0.55 * np.clip(r / (4.0 * scale_length_m), 0.0, 1.5)

    theta = theta + rng.normal(0.0, spread, count)

    # A fraction of the stars belong to a smooth disc with no arm
    # structure at all, plus a central bulge.
    smooth = rng.random(count) < 0.35
    theta = np.where(smooth, rng.random(count) * 2.0 * np.pi, theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Vertical structure: an exponential thin disc.
    z = rng.laplace(0.0, thickness * scale_length_m, count)

    return np.stack([x, y, z], axis=1)


def _invert_exponential_disc(u):
    """
    Invert the cumulative distribution of an exponential disc,
    F(x) = 1 - (1+x) e^-x, by Newton iteration on a good initial guess.

    Vectorised, converges in four passes to better than 1e-6.
    """

    u = np.clip(np.asarray(u, dtype=np.float64), 1.0e-9, 1.0 - 1.0e-9)

    # Initial guess from the asymptotic behaviour at both ends.
    x = np.where(u < 0.5, np.sqrt(2.0 * u), -np.log(np.maximum(1.0 - u, 1e-12)) + 1.2)

    for _ in range(6):
        f = 1.0 - (1.0 + x) * np.exp(-x) - u
        df = x * np.exp(-x)
        x = np.maximum(x - f / np.maximum(df, 1.0e-12), 1.0e-9)

    return x


def spheroid_positions(count, rng, effective_radius_m, flattening=0.75):
    """
    Star positions for an elliptical or a bulge, in metres.

    Radii follow a Hernquist profile, whose projection is very close to
    the de Vaucouleurs R^1/4 law that ellipticals actually obey.
    """

    # Hernquist: M(<r) = r^2/(r+a)^2, so r = a sqrt(u)/(1 - sqrt(u)).
    a = effective_radius_m / 1.8153

    u = rng.random(count) * 0.995
    root = np.sqrt(u)

    r = a * root / (1.0 - root)

    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    positions = direction * r[:, None]

    positions[:, 2] *= flattening

    return positions


def irregular_positions(count, rng, radius_m):
    """
    Clumpy, asymmetric star positions for an irregular galaxy.
    """

    n_clumps = 7

    centres = rng.normal(0.0, radius_m * 0.5, size=(n_clumps, 3))
    centres[:, 2] *= 0.4

    which = rng.integers(0, n_clumps, count)

    spread = radius_m * (0.12 + 0.3 * rng.random(count))[:, None]

    return centres[which] + rng.normal(size=(count, 3)) * spread


def galaxy_star_positions(count, rng, morphology, scale_radius_m):
    """
    Dispatch to the right spatial distribution for a morphology.
    """

    if morphology == MORPH_SPIRAL:
        arms = int(rng.integers(2, 5))
        return spiral_disc_positions(count, rng, scale_radius_m, arm_count=arms)

    if morphology in (MORPH_ELLIPTICAL, MORPH_DWARF):
        return spheroid_positions(count, rng, scale_radius_m)

    if morphology == MORPH_LENTICULAR:
        n_bulge = count // 3
        bulge = spheroid_positions(n_bulge, rng, scale_radius_m * 0.4)
        disc = spiral_disc_positions(
            count - n_bulge, rng, scale_radius_m, arm_count=2, pitch=0.6
        )
        return np.concatenate([bulge, disc], axis=0)

    return irregular_positions(count, rng, scale_radius_m)


def rotation_curve_velocity(radius_m, stellar_mass_solar, halo_mass_solar, expansion, cosmic_time):
    """
    Circular speed in m/s: disc plus halo added in quadrature.
    """

    from simulation_constants import G

    r = np.maximum(np.asarray(radius_m, dtype=np.float64), 1.0e15)

    m_star = np.asarray(stellar_mass_solar, dtype=np.float64) * SUN_MASS

    r_vir = virial_radius(halo_mass_solar, expansion, cosmic_time)
    c = concentration(halo_mass_solar, expansion.redshift(cosmic_time))

    from galaxy_pop.dark_matter_halos import nfw_circular_velocity

    v_halo = nfw_circular_velocity(r, halo_mass_solar, r_vir, c)

    # Disc contribution, treated as a point mass outside a few scale
    # lengths, which is accurate enough for a rotation curve.
    v_disc = np.sqrt(G * m_star / r)

    return np.sqrt(v_halo**2 + v_disc**2)
