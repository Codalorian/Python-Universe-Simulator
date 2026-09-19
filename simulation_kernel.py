"""
The simulation kernel.

Owns the clock, the expansion history, the galaxy catalogue, the
stellar population of whichever galaxy the observer is in, and the
observer themselves.

Two design choices are worth stating up front.

The state of the universe is a *function of cosmic time*, not an
accumulation of steps. Nothing integrates, so jumping a billion years
forward costs the same as advancing a single frame, running the clock
backwards to the Big Bang is exact, and there is no drift to correct.

The observer's position is stored relative to the galaxy they are
currently in, in parsecs, and rebased when they arrive somewhere new.
Absolute coordinates spanning the observable universe would need far
more precision than a float can offer; galaxy-relative coordinates
need almost none.
"""

import threading
import time as wallclock

from dataclasses import dataclass, field

import numpy as np

from simulation_constants import (
    C,
    YEAR,
    MYR,
    GYR,
    PARSEC,
    KPC,
    MPC,
    AU,
    RANDOM_SEED,
    GALAXY_COUNT,
    STAR_COUNT,
    UNIVERSE_RADIUS_MPC,
    MAX_FUTURE_TIME,
    DEFAULT_TIME_RATE,
    MAX_TIME_RATE,
)

from cosmology.cosmictime.time import CosmicTime, format_cosmic_time
from cosmology.expansion.scale_factor import ExpansionModel, PRESENT_AGE, describe_epoch
from cosmology.hubbleflow import HubbleFlow

from galaxy_pop.galaxy_formation import (
    generate_catalogue,
    MORPH_NAMES,
    MORPH_SPIRAL,
)
from galaxy_pop.mergers import resolve_mergers

from stellar_pop.stellar_evolution.population import (
    StellarPopulation,
    PHASE_REMNANT,
)

from stellar_pop.stellar_evolution.relations import (
    FATE_BLACK_HOLE,
    FATE_DIRECT_COLLAPSE,
)
from stellar_pop.supernovae import SN_TYPE_NAMES

from compact_objects.black_holes import eddington_luminosity_watts, shadow_radius

from compact_objects.quasars import (
    duty_cycle,
    bolometric_luminosity,
    eddington_ratio_distribution,
)

from planetary_systems.formation import generate_system

PARSEC_PER_MPC = 1.0e6

_NOTHING = np.empty(0, dtype=np.intp)

# Rebuilding the young stellar population re-sorts the whole event
# schedule, so it is rate limited in real time however fast the
# simulated clock is running.
RESEED_MIN_REAL_SECONDS = 2.0

# How much of the expected rebuild time to aim ahead by. Below one, so
# the clock lands just past the planned epoch rather than short of it.
RESEED_LEAD_FRACTION = 0.85

# Likewise for recomputing which black holes are currently quasars,
# and for resolving galaxy mergers.
QUASAR_MIN_REAL_SECONDS = 0.5
MERGER_MIN_REAL_SECONDS = 0.25

# Adopting a new host galaxy means building its whole stellar
# population, so it is rate limited however fast the observer is
# moving. Without this, crossing intergalactic space at speed queues
# a rebuild every frame.
REBASE_MIN_REAL_SECONDS = 2.0

# Travel speed is set by the distance to the nearest object, so that
# the same control works beside a planet and between galaxies. Both
# ends of that range have to be bounded: without an upper bound the
# observer accelerates away from everything for ever, since the
# further out they get the further away the nearest object is.
MIN_TRAVEL_REFERENCE_M = 1.0e9
MAX_TRAVEL_REFERENCE_M = 3.0e24

# Seconds to cross the reference distance at speed_setting 1.
TRAVEL_CROSSING_SECONDS = 4.0

# No single frame may move the observer more than this fraction of the
# reference distance. A long frame would otherwise fling them much
# further than a short one, putting them somewhere emptier and faster,
# which makes the next frame longer still.
MAX_STEP_FRACTION = 0.35


class BackgroundBuild:
    """
    One piece of heavy NumPy work, run off the main thread.

    Building a galaxy's stars takes half a second. Doing that inside a
    frame drops the frame rate to two for as long as it takes, which is
    exactly the sort of stall the user notices when flying between
    galaxies. NumPy releases the interpreter lock for the array
    operations that dominate here, so a plain thread genuinely overlaps
    with rendering.

    The work must only *read* simulation state. Installing the result
    is the caller's job, on the main thread.
    """

    def __init__(self, name, function, *args, **kwargs):
        self.name = name
        self.result = None
        self.error = None

        self.started = wallclock.perf_counter()

        self._thread = threading.Thread(
            target=self._run, args=(function, args, kwargs), daemon=True
        )

        self._thread.start()

    def _run(self, function, args, kwargs):
        try:
            self.result = function(*args, **kwargs)
        except BaseException as error:  # surfaced on the main thread
            self.error = error

    @property
    def done(self):
        return not self._thread.is_alive()

    @property
    def elapsed(self):
        return wallclock.perf_counter() - self.started


@dataclass(slots=True)
class SimulationEvent:
    event_type: str
    object_id: int
    cosmic_time: float
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{format_cosmic_time(self.cosmic_time)}] {self.event_type}"


@dataclass(slots=True)
class Observer:
    """
    Where the user is, and how fast they are going.

    Position is in parsecs relative to the centre of `galaxy_index`.
    """

    galaxy_index: int = 0
    position_pc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_c: float = 0.0
    speed_setting: float = 1.0

    def position_mpc(self, catalogue):
        centre = catalogue["position_mpc"][self.galaxy_index].astype(np.float64)

        return centre + self.position_pc / PARSEC_PER_MPC

    def distance_from_galaxy_centre_m(self):
        return float(np.linalg.norm(self.position_pc)) * PARSEC


class SimulationKernel:
    def __init__(self, seed=RANDOM_SEED, galaxy_count=GALAXY_COUNT, star_count=STAR_COUNT):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.expansion = ExpansionModel()
        self.hubble = HubbleFlow(self.expansion)

        self.time = CosmicTime(PRESENT_AGE)

        self.galaxy_count = int(galaxy_count)
        self.star_count = int(star_count)

        self.time_rate = DEFAULT_TIME_RATE
        self.paused = False

        self.events = []
        self.event_log = []

        self.catalogue = None
        self.population = None
        self.observer = Observer()

        self._system_cache = {}
        self._quasar_luminosity = None

        self._real_time_marks = {}

        self._pending_galaxy = None
        self._pending_reseed = None

        # Seeded with a plausible first guess; measured thereafter.
        self._last_reseed_seconds = 0.2

        # Set for one frame whenever the star population object is
        # swapped out, so the renderer knows to rebuild its buffers.
        self.population_replaced = False

        self.initialized = False

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def initialize(self, progress=None):
        if self.initialized:
            return

        def report(message):
            if progress is not None:
                progress(message)

        report("Solving the expansion history...")

        report(f"Generating {self.galaxy_count:,} galaxies...")

        self.catalogue = generate_catalogue(
            self.galaxy_count, self.rng, self.expansion, UNIVERSE_RADIUS_MPC
        )

        # Put the observer in a large spiral near the centre of the
        # volume, which is the analogue of standing in the Milky Way.
        self.observer.galaxy_index = self._pick_home_galaxy()

        report(f"Populating the home galaxy with {self.star_count:,} stars...")

        self.population = self._build_population(self.observer.galaxy_index)

        report("Lighting the quasars...")

        self._update_quasars()

        self.observer.position_pc = np.array([0.0, 0.0, -8_000.0])

        self.initialized = True

        report("Ready.")

    def _pick_home_galaxy(self):
        catalogue = self.catalogue

        spiral = catalogue["morphology"] == MORPH_SPIRAL

        distance = np.linalg.norm(catalogue["position_mpc"], axis=1)

        # A big spiral, close to the middle of the simulated volume.
        score = np.where(spiral, catalogue["stellar_mass"], 0.0) / (1.0 + distance)

        return int(np.argmax(score))

    def _build_population(self, galaxy_index):
        catalogue = self.catalogue

        morphology = int(catalogue["morphology"][galaxy_index])
        radius = float(catalogue["radius"][galaxy_index])

        # A deterministic per-galaxy stream, so the same galaxy always
        # has the same stars however many times you leave and return.
        rng = np.random.default_rng((self.seed * 1_000_003 + galaxy_index) % (2**63))

        return StellarPopulation(
            self.star_count,
            rng,
            self.expansion,
            radius,
            morphology,
        )

    def _build_population_at(self, galaxy_index, cosmic_time):
        """
        Build a galaxy's stars and bring them to a given epoch.

        Both halves run on the worker thread, so installing the result
        on the main thread is only a reference swap.
        """

        population = self._build_population(galaxy_index)

        population.evaluate_all(cosmic_time)

        return population

    # --------------------------------------------------------
    # The clock
    # --------------------------------------------------------

    @property
    def cosmic_time(self):
        return self.time.seconds

    def universe_age_gyr(self):
        return self.time.seconds / GYR

    def step(self, real_dt):
        """
        Advance one frame. Returns the indices of the stars whose
        appearance changed, ready to be handed to the renderer.
        """

        self.events.clear()

        # A paused universe is unchanging, so there is nothing to
        # recompute and nothing to re-upload.
        if self.paused or real_dt <= 0.0:
            return _NOTHING

        simulated_dt = real_dt * self.time_rate * YEAR

        target = min(self.time.seconds + simulated_dt, MAX_FUTURE_TIME)

        if target <= self.time.seconds:
            return _NOTHING

        self.time.jump_to(target)

        changed = self.population.advance_to(target)

        self._collect_supernovae(target)

        # The young stellar population turns over as the galaxy keeps
        # forming stars, so it is redrawn when the epoch has moved far
        # enough for the old draw to be meaningless.
        #
        # The test has to be against the wall clock as well as against
        # simulated time. At a trillion years per second three hundred
        # million years elapse in a third of a millisecond, and
        # redrawing the population every frame would cost more than
        # everything else in the simulator put together.
        if self.population.young_population_is_stale(
            target, 300.0 * MYR
        ) and self._real_seconds_since("reseed", RESEED_MIN_REAL_SECONDS):
            self._start_reseed(target)

        self._maybe_update_slow_systems(simulated_dt)

        finished = self.poll_background()

        return finished if finished is not None else changed

    def jump_to(self, cosmic_time):
        """
        Move the clock anywhere, including backwards.

        This is the one place a stall is acceptable: it is a deliberate
        action with a visible result, so it is done synchronously and
        the user sees the new epoch on the very next frame.
        """

        self.events.clear()

        target = float(np.clip(cosmic_time, 0.0, MAX_FUTURE_TIME))

        self.time.jump_to(target)

        changed = self.population.reseed_young_population(target)

        self._update_quasars()
        self._system_cache.clear()

        self._log(
            SimulationEvent(
                "time jump",
                -1,
                target,
                {"epoch": describe_epoch(target)},
            )
        )

        return changed

    def jump_by(self, seconds):
        return self.jump_to(self.time.seconds + seconds)

    def go_to_big_bang(self):
        return self.jump_to(0.0)

    def go_to_present(self):
        return self.jump_to(PRESENT_AGE)

    def cycle_time_rate(self, factor=10.0):
        self.time_rate *= factor

        if self.time_rate > MAX_TIME_RATE:
            self.time_rate = 1.0
        elif self.time_rate < 1.0:
            self.time_rate = MAX_TIME_RATE

        return self.time_rate

    # --------------------------------------------------------
    # Background work
    # --------------------------------------------------------

    def _start_reseed(self, cosmic_time):
        """
        Redraw the short-lived stellar population on a worker thread.

        The plan is built for where the clock is *expected to be* when
        the worker finishes, not for where it is now. At a billion
        years a second, the two hundred milliseconds a rebuild takes
        covers two hundred million years of stellar evolution, and
        catching up that much history in the frame the plan lands on
        costs more than the rebuild saved.

        The estimate is deliberately short, so the clock has usually
        just passed the planned epoch rather than not yet reached it:
        catching up a little is cheap, whereas a plan from the future
        would have to be thrown away.
        """

        if self._pending_reseed is not None or self._pending_galaxy is not None:
            return

        lead = 0.0

        if not self.paused:
            lead = (
                RESEED_LEAD_FRACTION
                * self._last_reseed_seconds
                * self.time_rate
                * YEAR
            )

        target = min(cosmic_time + lead, MAX_FUTURE_TIME)

        rng = np.random.default_rng(self.rng.integers(0, 2**62))

        self._pending_reseed = BackgroundBuild(
            "reseed", self.population.plan_young_reseed, target, rng
        )

    def _start_galaxy_build(self, galaxy_index):
        if self._pending_galaxy is not None:
            return

        self._pending_galaxy = BackgroundBuild(
            "galaxy", self._build_population_at, int(galaxy_index), self.time.seconds
        )

        self._pending_galaxy.galaxy_index = int(galaxy_index)

    def poll_background(self):
        """
        Install any finished background work. Returns the indices that
        need re-uploading, or None if nothing completed.
        """

        self.population_replaced = False

        pending = self._pending_galaxy

        if pending is not None and pending.done:
            self._pending_galaxy = None

            if pending.error is not None:
                self._log(
                    SimulationEvent(
                        "galaxy build failed",
                        pending.galaxy_index,
                        self.time.seconds,
                        {"error": type(pending.error).__name__},
                    )
                )
            else:
                self.population = pending.result

                # The clock kept running while the build was going, so
                # walk the new population forward over the events it
                # missed. That is the incremental path, not a full
                # re-evaluation, so it costs a fraction of a
                # millisecond rather than tens of them.
                if self.population.cosmic_time != self.time.seconds:
                    self.population.advance_to(self.time.seconds)

                self.population_replaced = True

                self._system_cache.clear()

                catalogue = self.catalogue
                index = pending.galaxy_index

                self._log(
                    SimulationEvent(
                        "arrived at galaxy",
                        index,
                        self.time.seconds,
                        {
                            "morphology": MORPH_NAMES[
                                int(catalogue["morphology"][index])
                            ],
                            "stellar_mass": float(catalogue["stellar_mass"][index]),
                        },
                    )
                )

                return np.arange(self.population.count)

        pending = self._pending_reseed

        if pending is not None and pending.done:
            self._pending_reseed = None

            self._last_reseed_seconds = pending.elapsed

            if pending.error is not None:
                return None

            # The estimate overshot and the plan is for an epoch that
            # has not arrived. Installing it would mean evaluating the
            # whole population backwards, which is exactly the stall
            # this machinery exists to avoid, so drop it and let the
            # next attempt aim better.
            if self.time.seconds < pending.result["reference_time"]:
                self._real_time_marks.pop("reseed", None)
                return None

            return self.population.apply_young_reseed(
                pending.result, self.time.seconds
            )

        return None

    @property
    def is_loading(self):
        return self._pending_galaxy is not None

    # --------------------------------------------------------
    # Slow subsystems
    # --------------------------------------------------------

    def _real_seconds_since(self, name, minimum):
        """
        True at most once per `minimum` seconds of wall clock.

        Several subsystems are naturally paced by simulated time, but
        simulated time can run twenty orders of magnitude faster than
        real time. Anything expensive needs both limits.
        """

        now = wallclock.perf_counter()

        last = self._real_time_marks.get(name)

        if last is not None and now - last < minimum:
            return False

        self._real_time_marks[name] = now

        return True

    def _maybe_update_slow_systems(self, simulated_dt):
        """
        Mergers and quasar activity change over hundreds of millions of
        years. Recomputing them every frame would be wasted work, so
        they are driven by accumulated simulated time instead.
        """

        self._merger_accumulator = getattr(self, "_merger_accumulator", 0.0) + simulated_dt

        if self._merger_accumulator < 20.0 * MYR:
            return

        if not self._real_seconds_since("mergers", MERGER_MIN_REAL_SECONDS):
            return

        dt = self._merger_accumulator
        self._merger_accumulator = 0.0

        merged = resolve_mergers(
            self.catalogue, dt, self.time.seconds, self.expansion, self.rng
        )

        for event in merged:
            self._log(
                SimulationEvent(
                    "galaxy merger" if event.is_major else "minor merger",
                    event.primary_index,
                    event.cosmic_time,
                    {"mass_ratio": round(event.mass_ratio, 3)},
                )
            )

        if self._real_seconds_since("quasars", QUASAR_MIN_REAL_SECONDS):
            self._update_quasars()

    def _update_quasars(self):
        """
        Decide which supermassive black holes are shining right now.
        """

        catalogue = self.catalogue

        z = float(self.expansion.redshift(self.time.seconds))

        masses = catalogue["smbh_mass"]

        active = self.rng.random(masses.shape) < duty_cycle(z)

        ratios = eddington_ratio_distribution(self.rng, masses.shape[0], z)

        self._quasar_luminosity = np.where(
            active & (catalogue["formation_time"] <= self.time.seconds),
            bolometric_luminosity(masses, ratios),
            0.0,
        )

        self.quasar_count = int((self._quasar_luminosity > 1.0e38).sum())

    @property
    def quasar_luminosity(self):
        return self._quasar_luminosity

    def _collect_supernovae(self, cosmic_time):
        """
        Turn newly exploded stars into events the UI can announce.
        """

        population = self.population

        exploding = population.active_supernovae(cosmic_time)

        if exploding.size == 0:
            self.supernova_indices = exploding
            return

        self.supernova_indices = exploding

        previous = getattr(self, "_announced_supernovae", set())

        fresh = [int(i) for i in exploding if int(i) not in previous]

        for index in fresh[:8]:
            self._log(
                SimulationEvent(
                    "supernova",
                    index,
                    cosmic_time,
                    {
                        "type": SN_TYPE_NAMES[int(population.sn_type[index])],
                        "mass_solar": float(population.mass[index]),
                    },
                )
            )

        self._announced_supernovae = set(int(i) for i in exploding)

    def _log(self, event):
        self.events.append(event)

        self.event_log.append(event)

        if len(self.event_log) > 256:
            del self.event_log[:128]

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    def force_supernova(self, star_index=None):
        """
        Detonate a star. With no argument, the nearest massive star
        that is still burning.
        """

        population = self.population

        if star_index is None:
            star_index = self.nearest_massive_star()

        if star_index is None:
            return None

        result = population.force_supernova(int(star_index), self.time.seconds)

        if result is None:
            return None

        self._log(
            SimulationEvent(
                "induced supernova",
                int(star_index),
                self.time.seconds,
                {
                    "type": result,
                    "mass_solar": float(population.mass[int(star_index)]),
                },
            )
        )

        return result

    def nearest_star(self):
        return self.population.nearest(self.observer.position_pc)

    def nearest_massive_star(self, minimum_mass=8.0):
        """
        The closest star that could actually undergo core collapse.
        """

        population = self.population

        candidates = population.massive_living_stars(minimum_mass)

        if candidates.size == 0:
            return None

        delta = population.position_pc[candidates] - self.observer.position_pc.astype(
            np.float32
        )

        return int(candidates[np.argmin(np.einsum("ij,ij->i", delta, delta))])

    def planetary_system(self, star_index):
        """
        The planets of one star, generated on demand and cached.
        """

        star_index = int(star_index)

        cached = self._system_cache.get(star_index)

        if cached is not None:
            return cached

        population = self.population

        system = generate_system(
            star_index,
            float(population.mass[star_index]),
            float(max(population.luminosity[star_index], 1.0e-6)),
            float(population.metallicity[star_index]),
            seed=self.seed,
        )

        if len(self._system_cache) > 64:
            self._system_cache.clear()

        self._system_cache[star_index] = system

        return system

    # --------------------------------------------------------
    # Black holes
    # --------------------------------------------------------

    @staticmethod
    def _deterministic(seed_value, count=3):
        """
        A stable handful of uniform deviates for one object, so its
        spin and orientation never change between visits.
        """

        rng = np.random.default_rng(int(seed_value) & 0x7FFFFFFF)

        return rng.random(count)

    def black_hole_spin_and_axis(self, identifier):
        """
        Spin magnitude and orientation for one hole.

        Accretion spins a black hole up towards the Thorne limit of
        a = 0.998, so an actively feeding hole is a fast rotator; the
        axis is fixed once and for all by the angular momentum of
        what formed it.
        """

        u = self._deterministic(identifier, 4)

        spin = 0.35 + 0.6 * u[0]

        theta = np.arccos(2.0 * u[1] - 1.0)
        phi = 2.0 * np.pi * u[2]

        axis = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi),
            ]
        )

        return float(spin), axis

    def stellar_black_hole_accretion(self, star_index):
        """
        Eddington ratio of a stellar-mass black hole.

        Most are isolated and utterly dark. A few per cent have a
        companion close enough to spill gas over its Roche lobe, and
        those are the X-ray binaries: bright, jetted microquasars.
        """

        u = self._deterministic(int(star_index) * 7919 + 13, 2)

        if u[0] > 0.06:
            return 0.0

        return float(10.0 ** (-2.2 + 2.2 * u[1]))

    def nearby_black_holes(self, limit=2, max_distance_m=2.0e17):
        """
        Stellar-mass black holes close enough to resolve.
        """

        population = self.population

        candidates = np.flatnonzero(
            (population.phase == PHASE_REMNANT)
            & (population.fate >= FATE_BLACK_HOLE)
            & (population.fate <= FATE_DIRECT_COLLAPSE)
        )

        if candidates.size == 0:
            return []

        # In float64. Positions are stored as float32, which is
        # plenty for placing a star inside a galaxy but quantises to
        # a couple of hundred AU - useless once the observer is a few
        # thousand kilometres from a black hole.
        delta = population.position_pc[candidates].astype(np.float64) - (
            self.observer.position_pc
        )

        distance_pc = np.sqrt(np.einsum("ij,ij->i", delta, delta))

        order = np.argsort(distance_pc)[:limit]

        found = []

        for slot in order:
            if distance_pc[slot] * PARSEC > max_distance_m:
                break

            index = int(candidates[slot])

            spin, axis = self.black_hole_spin_and_axis(index)

            found.append(
                {
                    "index": index,
                    "offset_m": delta[slot] * PARSEC,
                    "mass_solar": float(population.remnant_mass[index]),
                    "spin": spin,
                    "axis": axis,
                    "eddington_ratio": self.stellar_black_hole_accretion(index),
                }
            )

        return found

    def nearest_feeding_black_hole(self):
        """
        The closest stellar-mass black hole that is actually
        accreting, and so has a disc and jets to look at.
        """

        population = self.population

        candidates = np.flatnonzero(
            (population.phase == PHASE_REMNANT)
            & (population.fate >= FATE_BLACK_HOLE)
            & (population.fate <= FATE_DIRECT_COLLAPSE)
        )

        if candidates.size == 0:
            return None

        feeding = [
            int(index)
            for index in candidates
            if self.stellar_black_hole_accretion(int(index)) > 0.0
        ]

        if not feeding:
            return None

        delta = population.position_pc[feeding] - self.observer.position_pc.astype(
            np.float32
        )

        return feeding[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))]

    def brightest_active_nucleus(self):
        """
        The galaxy whose central black hole is currently shining
        hardest. Returns None when nothing is active, which is the
        usual state of affairs in the local universe.
        """

        if self._quasar_luminosity is None:
            return None

        index = int(np.argmax(self._quasar_luminosity))

        if self._quasar_luminosity[index] <= 0.0:
            return None

        return index

    def central_black_hole(self):
        """
        The supermassive black hole at the centre of the host galaxy.

        Its Eddington ratio comes from the quasar model, so it lights
        up during the epochs when quasars were common and sits dark
        today.
        """

        index = self.observer.galaxy_index

        mass = float(self.catalogue["smbh_mass"][index])

        spin, axis = self.black_hole_spin_and_axis(index + 1_000_003)

        luminosity = 0.0

        if self._quasar_luminosity is not None:
            luminosity = float(self._quasar_luminosity[index])

        eddington = 0.0

        if luminosity > 0.0:
            eddington = luminosity / float(eddington_luminosity_watts(mass))

        return {
            "offset_m": -self.observer.position_pc * PARSEC,
            "mass_solar": mass,
            "spin": spin,
            "axis": axis,
            "eddington_ratio": float(np.clip(eddington, 0.0, 2.0)),
        }

    # --------------------------------------------------------
    # Travel
    # --------------------------------------------------------

    def move_observer(self, direction, real_dt, boost=1.0):
        """
        Fly. `direction` is a unit vector in the observer's frame.

        Travel speed scales with how far away the nearest thing is, so
        the same control both crosses a hundred million light years and
        creeps up on a planet. The speed is reported honestly as a
        multiple of c: this is a map, not a spacecraft.
        """

        if real_dt <= 0.0:
            return

        reference = self.travel_reference_distance_m()

        speed_m_s = (
            reference
            / TRAVEL_CROSSING_SECONDS
            * self.observer.speed_setting
            * boost
        )

        step_m = speed_m_s * real_dt

        # Never cross more than a fraction of the reference distance in
        # one frame, whatever the frame took.
        step_m = min(step_m, MAX_STEP_FRACTION * reference)

        displacement_m = np.asarray(direction, dtype=np.float64) * step_m

        self.observer.position_pc = self.observer.position_pc + displacement_m / PARSEC

        self.observer.velocity_c = step_m / max(real_dt, 1.0e-6) / C

        self._maybe_rebase()

    def travel_reference_distance_m(self):
        """
        Distance to the nearest object, which sets the travel scale.

        Both stars and galaxies count. Inside a galaxy the nearest
        star is the thing you are manoeuvring around; once you leave,
        every star is far away and it is the next galaxy that sets the
        scale. Using only stars is what made travel accelerate without
        limit the moment the observer left the disc.
        """

        return float(
            np.clip(
                min(self.nearest_star_distance_m(), self.nearest_galaxy_distance_m()),
                MIN_TRAVEL_REFERENCE_M,
                MAX_TRAVEL_REFERENCE_M,
            )
        )

    def nearest_star_distance_m(self):
        population = self.population

        index = population.nearest(self.observer.position_pc)

        delta = population.position_pc[index].astype(np.float64) - (
            self.observer.position_pc
        )

        return float(np.linalg.norm(delta)) * PARSEC

    def nearest_galaxy_distance_m(self, tolerance=0.25):
        """
        Distance to the closest galaxy centre, in metres.

        Cached the same way the nearest star is: recomputed only once
        the observer has moved an appreciable fraction of the distance
        to it, so scanning the catalogue never lands in a frame more
        than occasionally.
        """

        here = self.observer.position_mpc(self.catalogue)

        cached = getattr(self, "_nearest_galaxy_cache", None)

        if cached is not None:
            index, origin, distance = cached

            if np.linalg.norm(here - origin) < tolerance * distance:
                return distance * MPC

        delta = self.catalogue["position_mpc"] - here.astype(np.float32)

        distances = np.einsum("ij,ij->i", delta, delta)

        distances[self.catalogue["halo_mass"] <= 0.0] = np.inf

        index = int(np.argmin(distances))

        distance = float(np.sqrt(distances[index]))

        self._nearest_galaxy_cache = (index, here.copy(), max(distance, 1.0e-9))

        return distance * MPC

    def _maybe_rebase(self):
        """
        If the observer has left their galaxy far behind, adopt the
        nearest galaxy as the new origin and build its stars.

        This is what keeps float precision usable: coordinates are
        always small numbers relative to something nearby.
        """

        catalogue = self.catalogue

        distance_mpc = float(np.linalg.norm(self.observer.position_pc)) / PARSEC_PER_MPC

        # Stay put until well outside the current galaxy's halo.
        if distance_mpc < 1.5:
            return None

        # Building a galaxy's stars is half a second of work, so do not
        # even look for a new host more often than that is worth.
        if not self._real_seconds_since("rebase", REBASE_MIN_REAL_SECONDS):
            return None

        if self._pending_galaxy is not None:
            return None

        here = self.observer.position_mpc(catalogue)

        delta = catalogue["position_mpc"] - here.astype(np.float32)

        distances = np.einsum("ij,ij->i", delta, delta)

        distances[catalogue["halo_mass"] <= 0.0] = np.inf

        nearest = int(np.argmin(distances))

        if nearest == self.observer.galaxy_index:
            return None

        # Rebase the coordinate origin immediately, which is cheap, and
        # build the new galaxy's stars in the background. Until they
        # arrive the observer is in intergalactic space with nothing
        # nearby to look at anyway.
        self.observer.galaxy_index = nearest

        self.observer.position_pc = (
            here - catalogue["position_mpc"][nearest].astype(np.float64)
        ) * PARSEC_PER_MPC

        self._start_galaxy_build(nearest)

        return nearest

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    def home_galaxy(self):
        catalogue = self.catalogue
        index = self.observer.galaxy_index

        return {
            "index": index,
            "morphology": MORPH_NAMES[int(catalogue["morphology"][index])],
            "stellar_mass": float(catalogue["stellar_mass"][index]),
            "halo_mass": float(catalogue["halo_mass"][index]),
            "radius_kpc": float(catalogue["radius"][index]) / KPC,
            "smbh_mass": float(catalogue["smbh_mass"][index]),
            "formation_redshift": float(catalogue["formation_redshift"][index]),
            "velocity_dispersion_km_s": float(catalogue["velocity_dispersion"][index]) / 1e3,
        }

    def cosmology_report(self):
        t = self.time.seconds

        return {
            "age": format_cosmic_time(t),
            "epoch": describe_epoch(t),
            "scale_factor": float(self.expansion.scale_factor(t)),
            "redshift": float(self.expansion.redshift(t)),
            "hubble_km_s_mpc": float(self.expansion.hubble_parameter(t)) * MPC / 1e3,
            "cmb_temperature": float(self.expansion.cmb_temperature(t)),
            "particle_horizon_gpc": float(self.expansion.particle_horizon) / MPC / 1e3,
            "deceleration": float(self.expansion.deceleration_parameter(t)),
        }

    def population_report(self):
        population = self.population

        phases = population.counts_by_phase()
        remnants = population.remnant_counts()

        return {
            "rendered stars": population.count,
            "main sequence": phases["main sequence"],
            "giants": phases["giant branch"],
            "remnants": phases["remnant"],
            "white dwarfs": remnants["white dwarfs"],
            "neutron stars": remnants["neutron stars"],
            "black holes": remnants["black holes"],
            "galaxies": int((self.catalogue["halo_mass"] > 0).sum()),
            "quasars": getattr(self, "quasar_count", 0),
        }
