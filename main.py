"""
UNIVERSE SIMULATOR
==================

A to-scale, physically grounded model of the universe from the Big
Bang to a billion years beyond the present day, flown in real time.

What is actually simulated
--------------------------
- Flat LambdaCDM expansion, integrated numerically from the Friedmann
  equation. Scale factor, redshift, comoving distance, horizon size
  and CMB temperature all come out of that one integration.
- The thermal history of the early universe, from the Planck epoch
  through nucleosynthesis and recombination to reionisation.
- A galaxy catalogue drawn from a cold dark matter halo mass function,
  with morphologies, sizes, black holes and formation epochs set by
  the scaling relations they obey in surveys, laid out along a cosmic
  web of filaments and voids.
- Stellar populations sampled from the Kroupa initial mass function
  with formation times following the cosmic star formation history.
  Luminosities, radii and temperatures are mutually consistent through
  the Stefan-Boltzmann law, and star colours are the true CIE colours
  of their Planck spectra.
- Stellar evolution: main sequence lifetimes from the nuclear
  timescale, the giant branch, and remnants - white dwarfs cooling by
  Mestel's law, neutron stars, and black holes.
- Supernovae: Types Ia, II and Ib/c, hypernovae and pair-instability
  explosions, with light curves powered by the Ni-56 to Co-56 to Fe-56
  decay chain.
- Quasars: Eddington-limited accretion onto supermassive black holes,
  with a duty cycle that peaks at cosmic noon and has faded by two
  orders of magnitude today.
- Planetary systems by core accretion, with Keplerian orbits solved
  exactly, including the relativistic precession of the perihelion.

Running it
----------
    pip install ursina numpy
    python3 main.py

Controls are listed on screen; press H to hide them.
"""

import argparse
import sys
import time as wallclock

import numpy as np

from ursina import (
    Ursina,
    Entity,
    Text,
    Vec2,
    Vec3,
    application,
    camera,
    color,
    held_keys,
    mouse,
    window,
)

import panda3d.core as p3d

from simulation_constants import (
    C,
    SUN_RADIUS,
    T_RECOMBINATION,
    RANDOM_SEED,
    STAR_COUNT,
    GALAXY_COUNT,
    YEAR,
    MYR,
    GYR,
    AU,
    PARSEC,
    MPC,
    SUN_RADIUS,
    EARTH_RADIUS,
)

from simulation_kernel import SimulationKernel

from cosmology.expansion.scale_factor import describe_epoch
from cosmology.bigbang import (
    radiation_temperature,
    cmb_glow_intensity,
)

from renderer import (
    format_distance,
    make_point_shader,
    LIGHT_YEAR,
)
from renderer.star_field import StarField
from renderer.galaxy_lod import GalaxyField
from renderer.nearby_objects import NearbyObjects, DETAIL_RANGE_METRES
from renderer.black_holes import jet_length_metres
from renderer.lensing import LensingPass, make_overlay_camera

from stellar_pop.stellar_evolution.blackbody import temperature_to_rgb, spectral_class
from stellar_pop.stellar_evolution.population import (
    PHASE_MAIN_SEQUENCE,
    PHASE_GIANT,
    PHASE_REMNANT,
    PHASE_NAMES,
)
from stellar_pop.stellar_evolution.relations import (
    FATE_NAMES,
    FATE_BLACK_HOLE,
    FATE_DIRECT_COLLAPSE,
)

from compact_objects.black_holes import shadow_radius, gravitational_radius
from compact_objects.quasars import jet_power


PARSEC_PER_MPC = 1.0e6

# How often the HUD text is rebuilt. Reformatting a dozen strings every
# frame is pure waste; ten times a second is smoother than the eye.
HUD_INTERVAL = 0.1

# Galaxies evolve on hundred-million-year timescales, so their vertex
# buffer does not need rebuilding at frame rate.
GALAXY_REFRESH_INTERVAL = 20.0 * MYR


# ============================================================
# Command line
# ============================================================

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="A to-scale simulation of the universe, from the Big Bang on.",
    )

    parser.add_argument(
        "--stars",
        type=int,
        default=STAR_COUNT,
        help="stars rendered for the host galaxy (default: %(default)s)",
    )

    parser.add_argument(
        "--galaxies",
        type=int,
        default=GALAXY_COUNT,
        help="galaxies in the catalogue (default: %(default)s)",
    )

    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="random seed"
    )

    parser.add_argument(
        "--benchmark",
        type=int,
        metavar="FRAMES",
        default=0,
        help="fly a fixed path for FRAMES frames, report the frame rate, and exit",
    )

    parser.add_argument(
        "--screenshot",
        metavar="PATH",
        default=None,
        help="render a few frames, save an image, and exit",
    )

    parser.add_argument(
        "--view",
        choices=(
            "disc",
            "galaxy",
            "cosmic",
            "system",
            "star",
            "blackhole",
            "quasar",
            "planet",
        ),
        default=None,
        help=(
            "start from a particular vantage point: inside the galactic "
            "disc, outside the whole galaxy, far enough out to see the "
            "cosmic web, or beside a planetary system"
        ),
    )

    parser.add_argument(
        "--standoff",
        type=float,
        metavar="RADII",
        default=3.0,
        help=(
            "how far to stand off in --view star (stellar radii) or "
            "--view blackhole / --view quasar (gravitational radii, "
            "default 120)"
        ),
    )

    parser.add_argument(
        "--age",
        type=float,
        metavar="GYR",
        default=None,
        help="start at this cosmic age in billions of years (0 is the Big Bang)",
    )

    parser.add_argument(
        "--memory-limit",
        type=float,
        metavar="MB",
        default=3072.0,
        help=(
            "abort if resident memory exceeds this (default: %(default)s MB; "
            "normal use is well under 600). 0 disables the check."
        ),
    )

    parser.add_argument(
        "--stress",
        type=float,
        metavar="SECONDS",
        default=0.0,
        help=(
            "hold full-boost travel for this long while logging memory, "
            "threads and frame time; for finding runaway behaviour"
        ),
    )

    parser.add_argument(
        "--selftest",
        action="store_true",
        help="press every control in turn, report any failures, and exit",
    )

    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="uncap the frame rate (use with --benchmark)",
    )

    parser.add_argument(
        "--windowed",
        type=str,
        metavar="WxH",
        default=None,
        help="window size, for example 1600x900",
    )

    return parser.parse_args(argv)


options = parse_arguments()


# ============================================================
# Startup
# ============================================================

def banner():
    print()
    print("=" * 62)
    print("   UNIVERSE SIMULATOR")
    print("=" * 62)
    print()


banner()

kernel = SimulationKernel(
    seed=options.seed,
    galaxy_count=options.galaxies,
    star_count=options.stars,
)

_start = wallclock.perf_counter()

kernel.initialize(progress=lambda message: print(f"   {message}"))

print(f"\n   Built in {wallclock.perf_counter() - _start:.2f} s")

report = kernel.population_report()

for name, value in report.items():
    print(f"   {name:>16}: {value:,}")

print()


# ============================================================
# Window
# ============================================================

_window_size = None

if options.windowed:
    try:
        width, height = (int(v) for v in options.windowed.lower().split("x"))
        _window_size = (width, height)
    except ValueError:
        print(f"   Ignoring unreadable window size {options.windowed!r}")

_headless = bool(
    options.benchmark or options.screenshot or options.selftest or options.stress
)

app = Ursina(
    title="Universe Simulator",
    borderless=False,
    fullscreen=False,
    vsync=not (options.no_vsync or options.benchmark),
    development_mode=False,
    **({"size": _window_size} if _window_size else {}),
)

window.color = color.black
window.exit_button.visible = False
window.fps_counter.enabled = False

camera.fov = 70
camera.position = Vec3(0, 0, 0)
camera.rotation = Vec3(0, 0, 0)

# The compression puts the edge of the observable universe at about
# 107 render units, so the far plane only has to clear that.
#
# The near plane cannot simply be made tiny to match. Past a near/far
# ratio of roughly a million the projection matrix loses the
# precision to separate the two and *everything* silently vanishes -
# which is what a near plane of 1e-5 against a far plane of 300 was
# doing to every solid body in the scene.
#
# 1.6e-4 keeps the ratio safe and corresponds to about fifty metres
# in front of the camera, which is closer than anything here is ever
# looked at. Depth precision itself does not matter: the star fields
# blend additively and the near-field bodies are drawn back to front,
# so depth testing is off throughout.
camera.clip_plane_near = 1.6e-4
camera.clip_plane_far = 160.0


# ============================================================
# Renderers
# ============================================================

point_shader = make_point_shader()

galaxy_field = GalaxyField(kernel.catalogue, kernel.expansion, shader=point_shader)
star_field = StarField(kernel.population, shader=point_shader)

nearby = NearbyObjects(overlay_discs=True)

# Gravitational lensing needs the scene in a texture before it can
# bend it, so this reroutes the whole 3D render through an offscreen
# buffer. The HUD is drawn separately and is unaffected.
lensing = LensingPass(
    application.base, application.base.cam, application.base.render
)

# Accretion discs bend themselves, in their own vertex shaders, and so
# are drawn over the finished frame instead of through the filter.
overlay_camera = make_overlay_camera(application.base)

# The primordial fireball, painted over the whole sky before the
# universe turns transparent.
fireball = Entity(
    parent=camera.ui,
    model="quad",
    scale=(4, 4),
    # Behind every HUD panel: before recombination the sky really is
    # an opaque wall of light, and the readout explaining why is the
    # one thing that still has to be legible through it.
    z=1.0,
    color=color.rgba32(0, 0, 0, 0),
    enabled=False,
)
# Ursina's UI layer ignores z ordering between siblings, so draw
# order is set explicitly by bin: the fireball first, every readable
# panel after it.
fireball.setBin("fixed", 0)
fireball.setDepthWrite(False)
fireball.setDepthTest(False)


# ============================================================
# Observer control
# ============================================================

class Flight:
    """
    A six-degree-of-freedom camera.

    The camera never actually moves: it sits at the origin and only
    turns. Travel changes the observer's position in the simulation,
    which the point shaders read as a uniform. That is what makes
    crossing a billion light years cost the same as standing still.
    """

    def __init__(self, kernel, capture_mouse=True):
        self.kernel = kernel

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.sensitivity = 40.0

        self.boost = 1.0

        mouse.locked = capture_mouse

    def basis(self):
        forward = camera.forward
        right = camera.right
        up = camera.up

        return (
            np.array([forward.x, forward.y, forward.z]),
            np.array([right.x, right.y, right.z]),
            np.array([up.x, up.y, up.z]),
        )

    def update(self, dt):
        if mouse.locked:
            self.yaw += mouse.velocity[0] * self.sensitivity * 25.0
            self.pitch -= mouse.velocity[1] * self.sensitivity * 25.0

            self.pitch = max(-89.5, min(89.5, self.pitch))

        roll_input = held_keys["e"] - held_keys["q"]
        self.roll += roll_input * 60.0 * dt

        camera.rotation = Vec3(self.pitch, self.yaw, self.roll)

        forward, right, up = self.basis()

        direction = (
            forward * (held_keys["w"] - held_keys["s"])
            + right * (held_keys["d"] - held_keys["a"])
            + up * (held_keys["r"] - held_keys["f"])
        )

        magnitude = float(np.linalg.norm(direction))

        if magnitude < 1.0e-6:
            self.kernel.observer.velocity_c = 0.0
            return

        direction = direction / magnitude

        boost = 10.0 if held_keys["shift"] else 1.0
        boost *= 0.08 if held_keys["control"] else 1.0

        self.kernel.move_observer(direction, dt, boost=boost)

    def look_at_universe_point(self, offset):
        """
        Aim at something, given its offset from the observer in
        universe axes.
        """

        offset = np.asarray(offset, dtype=np.float64)

        norm = float(np.linalg.norm(offset))

        if norm < 1.0e-12:
            return

        d = offset / norm

        self.yaw = float(np.degrees(np.arctan2(d[0], d[2])))
        self.pitch = float(np.degrees(np.arcsin(-np.clip(d[1], -1.0, 1.0))))


flight = Flight(kernel, capture_mouse=not _headless)


# ============================================================
# Heads-up display
# ============================================================

# Ursina's UI space runs from -0.5 to 0.5 vertically and from
# -aspect/2 to +aspect/2 horizontally, so the horizontal edges move
# with the window shape and have to be computed rather than guessed.
HUD_MARGIN = 0.015


def ui_edges():
    half_width = window.aspect_ratio * 0.5 - HUD_MARGIN

    return -half_width, half_width, 0.5 - HUD_MARGIN, -0.5 + HUD_MARGIN


def make_text(corner, size=0.7, shade=(236, 243, 255, 255)):
    """
    corner is one of "top left", "top right", "bottom left",
    "bottom right" or "mid right". Bottom-anchored panels grow upward,
    so a long block never runs off the screen.
    """

    left, right, top, bottom = ui_edges()

    origin_x = -0.5 if "left" in corner else 0.5
    x = left if "left" in corner else right

    if corner.startswith("bottom"):
        origin_y, y = -0.5, bottom
    elif corner.startswith("mid"):
        origin_y, y = 0.5, -0.05
    else:
        origin_y, y = 0.5, top

    element = Text(
        text="",
        parent=camera.ui,
        x=x,
        y=y,
        origin=(origin_x, origin_y),
        scale=size,
        color=color.rgba32(*shade),
        font="VeraMono.ttf",
    )

    element.hud_corner = corner

    return element


hud_left = make_text("top left")
hud_right = make_text("top right")
hud_target = make_text("bottom left")
hud_events = make_text("mid right", shade=(255, 196, 110, 255))

help_text = make_text("bottom right", size=0.62, shade=(160, 178, 208, 235))

HUD_PANELS = (hud_left, hud_right, hud_target, hud_events, help_text)


def layout_hud():
    """
    Reposition every panel. Called on startup and whenever the window
    changes shape.
    """

    left, right, top, bottom = ui_edges()

    for element in HUD_PANELS:
        corner = element.hud_corner

        element.x = left if "left" in corner else right

        if corner.startswith("bottom"):
            element.y = bottom
        elif corner.startswith("mid"):
            element.y = -0.05
        else:
            element.y = top


HELP_TEXT = (
    "W A S D  fly       R / F  up / down    Q / E  roll\n"
    "SHIFT  faster      CTRL  slower        MOUSE  look\n"
    "1 2 3 4  jump +1 / +10 / +100 Myr, +1 Gyr\n"
    "SHIFT + 1 2 3 4  the same, backwards\n"
    "B  Big Bang        P  present day      SPACE  pause\n"
    "T / G  time rate   X  detonate a star  N  next star\n"
    "C  go to a star    V  go to a galaxy   TAB  hide HUD\n"
    "M  go to a black hole                 K  go to a quasar\n"
    "H  hide this       ESC  release mouse  F10  quit"
)

help_text.text = HELP_TEXT

reticle = Text(
    text="+",
    parent=camera.ui,
    origin=(0, 0),
    scale=0.9,
    color=color.rgba32(160, 190, 230, 120),
)

# Colours below are 0-255 per channel, so they go through
# color.rgba32; Ursina's color.rgba takes 0-1 floats and silently
# clips anything larger to white.
#
# Panel colours for a dark sky and for a bright one. Before
# recombination the sky is a solid wall of light - correctly so, the
# universe was opaque - and that is exactly the moment the readout
# explaining why has to stay legible, so the HUD inverts.
HUD_DARK_SKY = {
    id(hud_left): (236, 243, 255, 255),
    id(hud_right): (236, 243, 255, 255),
    id(hud_target): (226, 236, 252, 255),
    id(hud_events): (255, 196, 110, 255),
    id(help_text): (160, 178, 208, 235),
}

HUD_BRIGHT_SKY = {
    id(hud_left): (20, 22, 30, 245),
    id(hud_right): (20, 22, 30, 245),
    id(hud_target): (20, 22, 30, 245),
    id(hud_events): (90, 30, 0, 245),
    id(help_text): (55, 60, 75, 225),
}


# Ursina's UI layer does not order siblings by z, so draw order is set
# explicitly by bin: the fireball first, every panel after it.
for _panel in HUD_PANELS:
    _panel.setBin("fixed", 100)

reticle.setBin("fixed", 100)


_hud_palette = {"current": HUD_DARK_SKY}


def set_hud_contrast(bright_sky):
    _hud_palette["current"] = HUD_BRIGHT_SKY if bright_sky else HUD_DARK_SKY

    for element in HUD_PANELS:
        write_panel(element, element.text)

    reticle.color = (
        color.rgba32(40, 40, 50, 160) if bright_sky else color.rgba32(160, 190, 230, 120)
    )


def write_panel(element, text):
    """
    Set a panel's text and its colour together.

    Ursina rebuilds a Text's geometry when its string changes, so the
    colour has to be reapplied alongside it or the panel reverts to
    whatever it was built with.
    """

    element.text = text
    element.color = color.rgba32(*_hud_palette["current"][id(element)])


layout_hud()

# ============================================================
# Selection and travel
# ============================================================

class Selection:
    def __init__(self):
        self.star_index = None
        self.system = None

    def select(self, index):
        self.star_index = None if index is None else int(index)

        self.system = (
            None if self.star_index is None else kernel.planetary_system(self.star_index)
        )

    def clear(self):
        self.star_index = None
        self.system = None


selection = Selection()


def observer_offset_to_star_m(index):
    """
    Vector from the observer to a star, in metres.
    """

    delta = kernel.population.position_pc[index].astype(np.float64)

    return (delta - kernel.observer.position_pc) * PARSEC


def match_time_rate_to_system(index):
    """
    Set the clock so the innermost planet takes about half a minute
    to go round.

    A million years a second is the right rate for watching stars
    live and die, and hopeless for watching a planet orbit: at that
    speed an inner world completes twenty million orbits a second
    and is simply a blur. Arriving somewhere should set a clock that
    suits what is there.
    """

    system = kernel.planetary_system(int(index))

    if not system.planets:
        return

    shortest = min(planet.period_seconds for planet in system.planets)

    kernel.time_rate = max(shortest / YEAR / 30.0, 1.0e-6)


def travel_to_star(index, standoff_au=800.0):
    """
    Place the observer just outside a star system and look at it.
    """

    index = int(index)

    target_pc = kernel.population.position_pc[index].astype(np.float64)

    standoff_pc = standoff_au * AU / PARSEC

    direction = np.array([0.6, 0.35, 0.72])
    direction /= np.linalg.norm(direction)

    kernel.observer.position_pc = target_pc - direction * standoff_pc

    selection.select(index)

    match_time_rate_to_system(index)

    flight.look_at_universe_point(direction)


def travel_to_galaxy(index=None):
    """
    Cross intergalactic space to another galaxy.
    """

    catalogue = kernel.catalogue

    if index is None:
        alive = np.flatnonzero(
            (catalogue["halo_mass"] > 0)
            & (catalogue["formation_time"] <= kernel.cosmic_time)
        )

        if alive.size == 0:
            return

        index = int(kernel.rng.choice(alive))

    kernel.observer.galaxy_index = int(index)

    radius_pc = float(catalogue["radius"][index]) / PARSEC

    kernel.observer.position_pc = np.array([0.0, 0.6, -3.0]) * max(radius_pc, 1.0e3)

    kernel.population = kernel._build_population_at(int(index), kernel.cosmic_time)

    rebuild_star_field()

    selection.clear()

    flight.look_at_universe_point(-kernel.observer.position_pc)


def rebuild_star_field():
    """
    Point the star field at a new host galaxy.
    """

    star_field.rebind(kernel.population)


def black_hole_standoff():
    """
    Viewing distance from a black hole, in gravitational radii.

    Scale-invariant: the same number frames a ten-solar-mass hole and
    a billion-solar-mass one identically, because every feature of a
    black hole scales with its mass.
    """

    if options.standoff and options.standoff != 3.0:
        return float(options.standoff)

    return 120.0


def go_to_planet():
    """
    Park a few planetary radii from the largest world of a nearby
    system, off to one side of the star so the terminator shows.
    """

    population = kernel.population

    for candidate in population.massive_living_stars(0.6)[:600]:
        system = kernel.planetary_system(int(candidate))

        if not system.planets:
            continue

        planet = max(system.planets, key=lambda p: p.radius_earth)

        slot = system.planets.index(planet)

        star_pc = population.position_pc[int(candidate)].astype(np.float64)

        local = system.positions_at(kernel.cosmic_time)[slot]

        planet_pc = star_pc + local / PARSEC

        radius_m = planet.radius_earth * EARTH_RADIUS

        # Stand off across the star-planet line, so roughly half the
        # disc is in daylight and half in shadow.
        away = local / max(float(np.linalg.norm(local)), 1.0e-9)

        side = np.cross(away, np.array([0.0, 1.0, 0.0]))
        side /= max(float(np.linalg.norm(side)), 1.0e-9)

        approach = away * 0.5 + side * 0.87
        approach /= np.linalg.norm(approach)

        # Same convention as travel_to_star: sit back along the
        # approach vector and look forward down it.
        kernel.observer.position_pc = planet_pc - approach * (
            4.0 * radius_m / PARSEC
        )

        selection.select(int(candidate))

        match_time_rate_to_system(int(candidate))

        # Standing four planetary radii off a world, the world itself
        # is moving tens of kilometres a second along its orbit: at
        # any rate fast enough to watch the orbit, it leaves the frame
        # before it can be looked at. Close inspection means a stopped
        # clock, until the observer can be put in the planet's own
        # frame of reference.
        kernel.paused = True

        flight.look_at_universe_point(approach)

        return

def go_to_black_hole():
    """
    Travel to the nearest feeding stellar-mass black hole, or failing
    that the nearest one of any kind.
    """

    target = kernel.nearest_feeding_black_hole()

    if target is None:
        holes = kernel.nearby_black_holes(limit=1, max_distance_m=1.0e22)

        if not holes:
            return

        target = holes[0]["index"]

    population = kernel.population

    mass = float(population.remnant_mass[target])

    # Far enough out that the disc and the inner jets both fit.
    standoff_m = black_hole_standoff() * float(gravitational_radius(mass))

    target_pc = population.position_pc[target].astype(np.float64)

    direction = np.array([0.35, 0.62, 0.70])
    direction /= np.linalg.norm(direction)

    kernel.observer.position_pc = target_pc - direction * (standoff_m / PARSEC)

    selection.select(int(target))

    flight.look_at_universe_point(direction)


def go_to_quasar():
    """
    Travel to the brightest active galactic nucleus in the catalogue.
    """

    index = kernel.brightest_active_nucleus()

    if index is None:
        return

    travel_to_galaxy(index)

    mass = float(kernel.catalogue["smbh_mass"][index])

    standoff_m = black_hole_standoff() * float(gravitational_radius(mass))

    direction = np.array([0.3, 0.55, 0.78])
    direction /= np.linalg.norm(direction)

    kernel.observer.position_pc = direction * (standoff_m / PARSEC) * -1.0

    flight.look_at_universe_point(-kernel.observer.position_pc)


def cycle_interesting_star():
    """
    Step through the most luminous stars nearby.
    """

    population = kernel.population

    alive = (population.phase == PHASE_MAIN_SEQUENCE) | (
        population.phase == PHASE_GIANT
    )

    candidates = np.flatnonzero(alive & (population.luminosity > 1_000.0))

    if candidates.size == 0:
        candidates = np.flatnonzero(alive)

    if candidates.size == 0:
        return

    delta = population.position_pc[candidates] - kernel.observer.position_pc.astype(
        np.float32
    )

    order = np.argsort(np.einsum("ij,ij->i", delta, delta))

    ranked = candidates[order[:64]]

    if selection.star_index is None or selection.star_index not in ranked:
        selection.select(int(ranked[0]))
    else:
        position = int(np.flatnonzero(ranked == selection.star_index)[0])
        selection.select(int(ranked[(position + 1) % len(ranked)]))

    flight.look_at_universe_point(observer_offset_to_star_m(selection.star_index))


# ============================================================
# Input
# ============================================================

show_hud = True
show_help = True

TIME_JUMPS = {
    "1": 1.0 * MYR,
    "2": 10.0 * MYR,
    "3": 100.0 * MYR,
    "4": 1.0 * GYR,
}


def apply_time_jump(seconds):
    kernel.jump_to(kernel.cosmic_time + seconds)

    on_universe_changed(full=True)


def on_universe_changed(full=False):
    """
    Called after anything that invalidates the whole render state.
    """

    star_field.refresh_all()

    galaxy_field.update(kernel.cosmic_time, kernel.quasar_luminosity)

    _galaxy_clock["last"] = kernel.cosmic_time
    _galaxy_clock["real"] = now_seconds()


def input(key):
    global show_hud, show_help

    back = held_keys["shift"]

    if key in TIME_JUMPS:
        apply_time_jump(-TIME_JUMPS[key] if back else TIME_JUMPS[key])

    elif key == "space":
        kernel.paused = not kernel.paused

    elif key == "b":
        kernel.go_to_big_bang()
        on_universe_changed(full=True)

    elif key == "p":
        kernel.go_to_present()
        on_universe_changed(full=True)

    elif key == "t":
        kernel.cycle_time_rate(10.0)

    elif key == "g":
        kernel.cycle_time_rate(0.1)

    elif key == "x":
        result = kernel.force_supernova(selection.star_index)

        if result is None and selection.star_index is not None:
            # The selected star cannot explode; fall back to the
            # nearest one that can.
            result = kernel.force_supernova(None)

    elif key == "n":
        cycle_interesting_star()

    elif key == "c":
        target = (
            selection.star_index
            if selection.star_index is not None
            else kernel.nearest_star()
        )

        if target is not None:
            travel_to_star(target)

    elif key == "v":
        travel_to_galaxy()

    elif key == "m":
        go_to_black_hole()

    elif key == "k":
        go_to_quasar()

    elif key == "tab":
        show_hud = not show_hud

        for element in HUD_PANELS:
            element.enabled = show_hud

        reticle.enabled = show_hud

        help_text.enabled = show_hud and show_help

    elif key == "h":
        show_help = not show_help
        help_text.enabled = show_hud and show_help

    elif key == "escape":
        mouse.locked = not mouse.locked

    elif key == "f10":
        application.quit()


# ============================================================
# Per-frame update
# ============================================================

class MemoryGuard:
    """
    Stop the simulation before it can take the machine down with it.

    Steady-state use sits around half a gigabyte. Anything that climbs
    past a few gigabytes is a runaway, and a runaway in a real-time
    loop reaches swap in under a minute - which locks up the whole
    desktop, not just this process. Better to exit with an explanation
    than to become unkillable.
    """

    CHECK_INTERVAL = 1.0

    def __init__(self, limit_mb):
        self.limit_mb = float(limit_mb)
        self.next_check = 0.0
        self.baseline = self.resident_mb()

    @staticmethod
    def resident_mb():
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass

        return 0.0

    def check(self, now):
        if self.limit_mb <= 0 or now < self.next_check:
            return

        self.next_check = now + self.CHECK_INTERVAL

        current = self.resident_mb()

        if current < self.limit_mb:
            return

        print()
        print("   " + "!" * 56)
        print(f"   Memory guard tripped: {current:,.0f} MB in use, limit "
              f"{self.limit_mb:,.0f} MB.")
        print(f"   Started at {self.baseline:,.0f} MB. Something is leaking;")
        print("   stopping now so the machine stays responsive.")
        print("   Raise or disable it with --memory-limit.")
        print("   " + "!" * 56)
        print()

        application.quit()


class FrameTimer:
    """
    Per-section timings, so the benchmark can say *what* was slow
    rather than only that something was.
    """

    SECTIONS = ("fly", "physics", "upload", "camera", "galaxies", "nearby", "hud")

    def __init__(self):
        self.totals = {name: 0.0 for name in self.SECTIONS}
        self.peaks = {name: 0.0 for name in self.SECTIONS}
        self.frames = 0
        self._mark = 0.0

    def start(self):
        self._mark = wallclock.perf_counter()

    def lap(self, name):
        now = wallclock.perf_counter()
        elapsed = now - self._mark
        self._mark = now

        self.totals[name] += elapsed
        self.peaks[name] = max(self.peaks[name], elapsed)

        return elapsed

    def end_frame(self):
        self.frames += 1

    def report(self):
        if not self.frames:
            return []

        return [
            (name, self.totals[name] / self.frames * 1000.0, self.peaks[name] * 1000.0)
            for name in self.SECTIONS
        ]


frame_timer = FrameTimer()

memory_guard = MemoryGuard(options.memory_limit)

_layout = {"aspect": window.aspect_ratio}

_sky = {"bright": False}

_hud_clock = {"next": 0.0, "panel": 0}
_galaxy_clock = {"last": kernel.cosmic_time, "real": 0.0}
_frame_times = []


def now_seconds():
    return wallclock.perf_counter()


def update():
    from ursina import time as ursina_time

    dt = min(ursina_time.dt, 0.1)

    frame_start = wallclock.perf_counter()

    if window.aspect_ratio != _layout["aspect"]:
        _layout["aspect"] = window.aspect_ratio
        layout_hud()
        star_field.cloud.set_window_size()
        galaxy_field.cloud.set_window_size()

    frame_timer.start()

    # ---- Fly ----
    flight.update(dt)

    frame_timer.lap("fly")

    # ---- Physics ----
    changed = kernel.step(dt)

    frame_timer.lap("physics")

    if kernel.population_replaced:
        # A background build finished and swapped in a new galaxy's
        # stars, so the vertex buffers have to be repointed.
        rebuild_star_field()
    elif len(changed):
        star_field.refresh_rows(changed)

    frame_timer.lap("upload")

    # ---- Camera uniforms: the entire cost of moving through space ----
    observer = kernel.observer

    star_field.set_camera_parsecs(observer.position_pc)

    galaxy_field.set_camera_mpc(
        observer.position_mpc(kernel.catalogue), kernel.cosmic_time
    )

    galaxy_field.set_host(observer.galaxy_index)

    frame_timer.lap("camera")

    # ---- Galaxies, on their own slower clock ----
    #
    # Galaxies appear and brighten over hundreds of millions of years,
    # but simulated time can outrun the wall clock by twenty orders of
    # magnitude, so the refresh is limited by both.
    if (
        abs(kernel.cosmic_time - _galaxy_clock["last"]) > GALAXY_REFRESH_INTERVAL
        and now_seconds() - _galaxy_clock["real"] > 0.2
    ):
        galaxy_field.update(kernel.cosmic_time, kernel.quasar_luminosity)
        _galaxy_clock["last"] = kernel.cosmic_time
        _galaxy_clock["real"] = now_seconds()

    frame_timer.lap("galaxies")

    # ---- Things close enough to resolve ----
    draw_nearby()

    # ---- Early universe ----
    draw_fireball()

    lensing.update(
        nearby.lens_sources(),
        application.base.cam,
        camera.lens,
        window.aspect_ratio,
        application.base.render,
    )

    frame_timer.lap("nearby")

    # ---- HUD ----
    now = wallclock.perf_counter()

    memory_guard.check(now)

    _frame_times.append(now - frame_start)

    if len(_frame_times) > 120:
        del _frame_times[:60]

    # Rebuilding a text panel regenerates its geometry, so the four
    # panels take turns rather than all landing on the same frame.
    if show_hud and now >= _hud_clock["next"]:
        _hud_clock["next"] = now + HUD_INTERVAL * 0.25

        update_hud_panel(_hud_clock["panel"])

        _hud_clock["panel"] = (_hud_clock["panel"] + 1) % 4

    frame_timer.lap("hud")
    frame_timer.end_frame()

    # ---- Automated modes ----
    if benchmark is not None:
        benchmark.tick(dt, true_dt=ursina_time.dt)

    if screenshot is not None:
        screenshot.tick()

    if selftest is not None:
        selftest.tick()

    if stress is not None:
        stress.tick(dt)


def draw_nearby():
    """
    Draw everything close enough to resolve: the star system in front
    of the observer, and any black hole making itself known.
    """

    nearby.begin_frame()

    population = kernel.population
    cosmic_time = kernel.cosmic_time

    index = selection.star_index

    if index is None:
        index = population.nearest(kernel.observer.position_pc)

    offset = observer_offset_to_star_m(index)

    distance = float(np.linalg.norm(offset))

    is_black_hole = population.phase[index] == PHASE_REMNANT and (
        FATE_BLACK_HOLE <= population.fate[index] <= FATE_DIRECT_COLLAPSE
    )

    if distance < DETAIL_RANGE_METRES and not is_black_hole:
        draw_star_system(index, offset, distance, cosmic_time)

    draw_black_holes(cosmic_time)

    nearby.end_frame()


def draw_star_system(index, offset, distance, cosmic_time):
    """
    One star and its planets, each lit by that star.
    """

    population = kernel.population

    temperature = float(population.temperature[index])
    luminosity = float(population.luminosity[index])

    radius_m = float(population.radius[index]) * SUN_RADIUS

    nearby.add_star(offset, radius_m, temperature, luminosity)

    system = kernel.planetary_system(index)

    if not system.planets:
        return

    star_rgb = temperature_to_rgb(max(temperature, 300.0))

    positions = system.positions_at(cosmic_time)

    for planet, local in zip(system.planets, positions):
        # Insolation relative to Earth's: L / a^2 in solar units.
        insolation = luminosity / max(planet.semi_major_axis_au**2, 1.0e-6)

        nearby.add_planet(
            offset + local,
            planet.radius_earth * EARTH_RADIUS,
            planet.planet_type,
            star_offset_metres=offset,
            star_rgb=star_rgb,
            star_flux=insolation**0.35,
        )

        nearby.add_orbit_ring(
            offset,
            planet.semi_major_axis_au * AU,
            planet.inclination,
            planet.longitude_of_node,
        )


def draw_black_holes(cosmic_time):
    """
    The host galaxy's central hole, plus any stellar-mass hole near
    enough to see.

    A quiescent hole is drawn only once its shadow is big enough to
    make out. A feeding one is drawn from much further away, because
    its jets are kiloparsecs long and are meant to be seen.
    """

    candidates = []

    central = kernel.central_black_hole()

    candidates.append(central)
    candidates.extend(kernel.nearby_black_holes(limit=2, max_distance_m=4.0e19))

    for hole in candidates:
        offset = np.asarray(hole["offset_m"], dtype=np.float64)

        distance = float(np.linalg.norm(offset))

        if distance < 1.0:
            continue

        feeding = hole["eddington_ratio"] > 0.0

        extent = (
            jet_length_metres(
                float(
                    jet_power(
                        hole["mass_solar"],
                        hole["eddington_ratio"],
                        max(hole["spin"], 0.3),
                    )
                )
            )
            if feeding
            else float(shadow_radius(hole["mass_solar"]))
        )

        # Skip anything subtending less than about a pixel.
        if extent / distance < 0.0008:
            continue

        nearby.add_black_hole(
            offset,
            hole["mass_solar"],
            spin=hole["spin"],
            eddington_ratio=hole["eddington_ratio"],
            spin_axis=hole["axis"],
            cosmic_time=cosmic_time,
        )


def draw_fireball():
    """
    Before recombination the universe is an opaque glowing plasma, so
    the sky is filled with its blackbody colour.
    """

    t = kernel.cosmic_time

    intensity = cmb_glow_intensity(t)

    bright = intensity > 0.25

    if bright != _sky["bright"]:
        _sky["bright"] = bright
        set_hud_contrast(bright)

    if intensity <= 0.001:
        if fireball.enabled:
            fireball.enabled = False
        return

    fireball.enabled = True

    temperature = float(radiation_temperature(max(t, 1.0)))

    rgb = temperature_to_rgb(min(temperature, 40_000.0))

    fireball.color = color.rgba32(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), int(intensity * 255)
    )


def update_hud_panel(panel):
    if panel == 0:
        update_clock_panel()
    elif panel == 1:
        update_galaxy_panel()
    elif panel == 2:
        write_panel(hud_target, describe_target())
    else:
        update_event_panel()


def update_clock_panel():
    cosmology = kernel.cosmology_report()
    observer = kernel.observer

    mean_frame = float(np.mean(_frame_times)) if _frame_times else 0.0

    from ursina import time as ursina_time

    fps = 1.0 / max(ursina_time.dt, 1.0e-6)

    rate = kernel.time_rate

    write_panel(hud_left, (
        f"AGE         {cosmology['age']}\n"
        f"EPOCH       {cosmology['epoch']}\n"
        f"REDSHIFT    z = {cosmology['redshift']:+.4f}\n"
        f"SCALE       a = {cosmology['scale_factor']:.5f}\n"
        f"HUBBLE      {cosmology['hubble_km_s_mpc']:,.1f} km/s/Mpc\n"
        f"CMB         {cosmology['cmb_temperature']:,.3f} K"
        f"{'  (opaque plasma)' if kernel.cosmic_time < T_RECOMBINATION else ''}\n"
        f"HORIZON     {cosmology['particle_horizon_gpc']:.2f} Gpc\n"
        f"\n"
        f"TIME RATE   {rate:,.0f} yr/s{'   [PAUSED]' if kernel.paused else ''}\n"
        f"SPEED       {observer.velocity_c:,.0f} c\n"
        f"FRAME       {fps:5.1f} fps   sim {mean_frame * 1000.0:4.1f} ms"
    ))


def update_galaxy_panel():
    galaxy = kernel.home_galaxy()
    stats = kernel.population_report()
    observer = kernel.observer

    write_panel(hud_right, (
        f"HOST GALAXY  {galaxy['morphology']} #{galaxy['index']}\n"
        f"STELLAR MASS {galaxy['stellar_mass']:.3e} Msun\n"
        f"HALO MASS    {galaxy['halo_mass']:.3e} Msun\n"
        f"CENTRAL BH   {galaxy['smbh_mass']:.3e} Msun\n"
        f"FORMED AT    z = {galaxy['formation_redshift']:.2f}\n"
        f"FROM CENTRE  {format_distance(observer.distance_from_galaxy_centre_m())}\n"
        f"\n"
        f"STARS DRAWN  {stats['rendered stars']:,}\n"
        f"MAIN SEQ     {stats['main sequence']:,}\n"
        f"GIANTS       {stats['giants']:,}\n"
        f"WHITE DWARFS {stats['white dwarfs']:,}\n"
        f"NEUTRON STARS{stats['neutron stars']:,}\n"
        f"BLACK HOLES  {stats['black holes']:,}\n"
        f"GALAXIES     {stats['galaxies']:,}\n"
        f"QUASARS      {stats['quasars']:,}"
    ))


def update_event_panel():
    recent = kernel.event_log[-6:]

    write_panel(hud_events, "\n".join(
        f"{event.event_type}: "
        + ", ".join(
            f"{name} {_short(value)}" for name, value in list(event.data.items())[:2]
        )
        for event in reversed(recent)
    ))


def _short(value):
    """
    Event values come straight from the simulation, so trim the ones
    that would otherwise print seventeen significant figures.
    """

    if isinstance(value, float):
        return f"{value:,.4g}"

    return value


def describe_target():
    population = kernel.population

    index = selection.star_index

    if index is None:
        index = population.nearest(kernel.observer.position_pc)
        label = "NEAREST STAR"
    else:
        label = "SELECTED STAR"

    offset = observer_offset_to_star_m(index)
    distance = float(np.linalg.norm(offset))

    phase = int(population.phase[index])

    lines = [
        f"{label} #{index}",
        f"DISTANCE     {format_distance(distance)}",
        f"MASS         {population.mass[index]:.3f} Msun",
        f"CLASS        {spectral_class(population.temperature[index])}"
        f"   {PHASE_NAMES[phase]}",
        f"LUMINOSITY   {population.luminosity[index]:.4g} Lsun",
        f"RADIUS       {population.radius[index]:.4g} Rsun",
        f"TEMPERATURE  {population.temperature[index]:,.0f} K",
        f"METALLICITY  {population.metallicity[index]:.5f}",
        f"BORN         {population.formation_time[index] / GYR:.3f} Gyr",
        f"FATE         {FATE_NAMES[int(population.fate[index])]}",
    ]

    system = kernel.planetary_system(index)

    if system.planets:
        lines.append(f"PLANETS      {len(system.planets)}")

        for planet in system.planets[:5]:
            lines.append(
                f"  {planet.name:<4} {planet.semi_major_axis_au:7.3f} AU "
                f"{planet.mass_earth:8.2f} Me  {planet.equilibrium_temperature:5.0f} K  "
                f"{planet.type_name}"
            )

        habitable = system.habitable_planets

        if habitable:
            lines.append(
                "  habitable: " + ", ".join(p.name for p in habitable)
            )
    else:
        lines.append("PLANETS      none")

    return "\n".join(lines)


# ============================================================
# Starting viewpoints
# ============================================================

def apply_view(name):
    """
    Place the observer somewhere worth looking from.
    """

    catalogue = kernel.catalogue
    index = kernel.observer.galaxy_index

    radius_pc = float(catalogue["radius"][index]) / PARSEC

    if name == "galaxy":
        kernel.observer.position_pc = np.array([0.0, 6.0, -9.0]) * radius_pc
        flight.look_at_universe_point(-kernel.observer.position_pc)

    elif name == "cosmic":
        # Far enough out that individual galaxies are points and the
        # filaments and voids they trace become the subject.
        kernel.observer.position_pc = np.array([0.0, 0.35, -1.0]) * 6.0e8
        flight.look_at_universe_point(-kernel.observer.position_pc)

    elif name == "star":
        target = kernel.nearest_star()

        radius_m = float(kernel.population.radius[target]) * SUN_RADIUS

        travel_to_star(target, standoff_au=float(options.standoff) * radius_m / AU)

    elif name == "planet":
        go_to_planet()

    elif name == "blackhole":
        go_to_black_hole()

    elif name == "quasar":
        go_to_quasar()

    elif name == "system":
        target = None

        for candidate in kernel.population.massive_living_stars(1.0)[:400]:
            if kernel.planetary_system(int(candidate)).planets:
                target = int(candidate)
                break

        if target is None:
            target = kernel.nearest_star()

        travel_to_star(target, standoff_au=40.0)

    else:
        kernel.observer.position_pc = np.array([0.0, 0.25, -1.0]) * radius_pc * 0.9
        flight.look_at_universe_point(-kernel.observer.position_pc)


# ============================================================
# Benchmark and screenshot modes
# ============================================================

class Benchmark:
    """
    Fly a fixed path through the simulation and time it.

    The path deliberately exercises the expensive cases: a fast
    traverse, a full-rate time run, and a stretch sitting inside a
    planetary system where the near-field renderer is busiest.
    """

    WARMUP = 30

    def __init__(self, frames):
        self.frames = int(frames)
        self.index = 0
        self.samples = []
        self.cpu_samples = []

    def tick(self, dt, true_dt=None):
        self.index += 1

        stage = self.index / max(self.frames, 1)

        flight.yaw += 24.0 * dt
        flight.pitch = 12.0 * np.sin(self.index * 0.01)

        forward, right, up = flight.basis()

        kernel.move_observer(forward, dt, boost=8.0 if stage < 0.5 else 1.0)

        if self.index == int(self.frames * 0.34):
            kernel.time_rate = 1.0e9

        if self.index == int(self.frames * 0.67):
            kernel.time_rate = 1.0e12

        if self.index > self.WARMUP:
            self.samples.append(true_dt if true_dt is not None else dt)
            self.cpu_samples.append(_frame_times[-1] if _frame_times else 0.0)

        if self.index >= self.frames:
            self.report()
            application.quit()

    def report(self):
        if not self.samples:
            print("   Benchmark produced no samples.")
            return

        times = np.array(self.samples) * 1000.0

        print()
        print("   " + "-" * 56)
        print(f"   BENCHMARK  {len(times)} frames")
        print("   " + "-" * 56)
        print(f"   stars     {kernel.population.count:,}")
        print(f"   galaxies  {kernel.catalogue['halo_mass'].size:,}")
        print()
        print(f"   mean      {times.mean():7.2f} ms   {1000.0 / times.mean():7.1f} fps")
        print(f"   median    {np.median(times):7.2f} ms   {1000.0 / np.median(times):7.1f} fps")
        print(f"   95th pct  {np.percentile(times, 95):7.2f} ms   "
              f"{1000.0 / np.percentile(times, 95):7.1f} fps")
        print(f"   worst     {times.max():7.2f} ms   {1000.0 / times.max():7.1f} fps")
        print()
        cpu = np.array(self.cpu_samples) * 1000.0

        print(f"   frames under 16.7 ms: {100.0 * (times < 16.7).mean():.1f}%")
        print()
        print(f"   of which CPU (simulation + scene update):")
        print(f"     mean    {cpu.mean():7.2f} ms      median {np.median(cpu):7.2f} ms")
        print(f"   remainder is GPU draw and buffer swap:")
        print(f"     mean    {times.mean() - cpu.mean():7.2f} ms")
        print()
        rebases = sum(
            1 for event in kernel.event_log if event.event_type == "arrived at galaxy"
        )

        supernovae = sum(
            1 for event in kernel.event_log if "supernova" in event.event_type
        )

        print(f"   galaxy arrivals during run: {rebases}    supernovae: {supernovae}")
        print()
        print("   CPU breakdown            mean ms    worst ms")

        for name, mean_ms, peak_ms in frame_timer.report():
            print(f"     {name:<20} {mean_ms:7.3f}     {peak_ms:7.2f}")

        print("   " + "-" * 56)
        print()


benchmark = Benchmark(options.benchmark) if options.benchmark else None


class SelfTest:
    """
    Drive every control and report anything that raises.

    The physics has unit coverage; this covers the other half, where
    a key press reaches into the renderer and the simulation at the
    same time. Each key gets its own frame so the consequences are
    actually drawn before the next one is pressed.
    """

    KEYS = (
        "space", "space",
        "1", "2", "3", "4",
        "b", "p",
        "t", "t", "g",
        "n", "n", "x", "c", "x",
        "v", "n", "x",
        "tab", "tab", "h", "h",
        "escape", "escape",
    )

    FRAMES_PER_KEY = 3

    def __init__(self):
        self.index = 0
        self.failures = []
        self.pressed = 0

    def tick(self):
        self.index += 1

        if self.index % self.FRAMES_PER_KEY:
            return

        step = self.index // self.FRAMES_PER_KEY - 1

        if step >= len(self.KEYS):
            self.report()
            application.quit()
            return

        key = self.KEYS[step]

        try:
            input(key)
            self.pressed += 1
        except Exception as error:
            self.failures.append(f"{key}: {type(error).__name__}: {error}")

    def report(self):
        print()
        print("   " + "-" * 56)
        print(f"   SELF TEST  {self.pressed}/{len(self.KEYS)} controls exercised")

        for failure in self.failures:
            print(f"     FAILED  {failure}")

        if not self.failures:
            print("   all controls responded without error")

        print(f"   universe age now {kernel.universe_age_gyr():.4f} Gyr")
        print(f"   events logged: {len(kernel.event_log)}")
        print("   " + "-" * 56)
        print()


selftest = SelfTest() if options.selftest else None


class Stress:
    """
    Fly flat out and report what grows.

    A simulation that merely runs slowly is a different problem from
    one that grows without bound, and only the second kind takes a
    machine down with it. This watches the quantities that could:
    resident memory, live threads, frame time, and how far and how
    fast the observer has got.
    """

    def __init__(self, seconds):
        self.deadline = wallclock.perf_counter() + float(seconds)
        self.next_sample = 0.0
        self.samples = []
        self.worst_frame = 0.0

    @staticmethod
    def resident_mb():
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass

        return float("nan")

    def tick(self, dt):
        import threading

        self.worst_frame = max(self.worst_frame, dt)

        forward, _, _ = flight.basis()

        kernel.move_observer(forward, dt, boost=10.0)

        flight.yaw += 8.0 * dt

        now = wallclock.perf_counter()

        if now >= self.next_sample:
            self.next_sample = now + 1.0

            observer = kernel.observer

            self.samples.append(
                (
                    self.resident_mb(),
                    threading.active_count(),
                    self.worst_frame * 1000.0,
                    float(np.linalg.norm(observer.position_pc)) / 1.0e6,
                    observer.velocity_c,
                    kernel.observer.galaxy_index,
                )
            )

            self.worst_frame = 0.0

        if now >= self.deadline:
            self.report()
            application.quit()

    def report(self):
        print()
        print("   " + "-" * 72)
        print("   STRESS: full-boost travel")
        print("   " + "-" * 72)
        print(
            f"   {'t':>3} {'RSS MB':>9} {'threads':>8} {'worst ms':>9} "
            f"{'from host Mpc':>14} {'speed / c':>13} {'galaxy':>8}"
        )

        for index, row in enumerate(self.samples):
            rss, threads, worst, distance, speed, galaxy = row
            print(
                f"   {index:>3} {rss:9.1f} {threads:8d} {worst:9.1f} "
                f"{distance:14.4g} {speed:13.3e} {galaxy:8d}"
            )

        if self.samples:
            first, last = self.samples[0], self.samples[-1]
            print()
            print(f"   memory drift: {last[0] - first[0]:+.1f} MB over {len(self.samples)} s")
            print(f"   thread drift: {last[1] - first[1]:+d}")

        print("   " + "-" * 72)
        print()


stress = Stress(options.stress) if options.stress else None


class Screenshot:
    def __init__(self, path, settle=20):
        self.path = path
        self.settle = settle
        self.index = 0

    def tick(self):
        self.index += 1

        if self.index < self.settle:
            return

        from ursina import application as ursina_application

        ursina_application.base.win.saveScreenshot(
            p3d.Filename.fromOsSpecific(self.path)
        )

        print(f"   Saved {self.path}")

        application.quit()


screenshot = Screenshot(options.screenshot) if options.screenshot else None


# ============================================================
# Go
# ============================================================

if options.age is not None:
    kernel.jump_to(options.age * GYR)

if options.view:
    apply_view(options.view)

if not _headless:
    print("   Controls are on screen. ESC releases the mouse, F10 quits.\n")

on_universe_changed(full=True)

app.run()
