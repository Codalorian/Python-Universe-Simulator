"""
The stellar population, held as a struct of arrays.

Two ideas make a quarter of a million individually evolving stars fit
inside a 60 Hz frame.

The first is that a star's state is a *pure function of cosmic time*.
Nothing is integrated: given t, a star's age is t minus its formation
time, and its phase follows from comparing that age against two fixed
timescales. So time travel is exact and instant in either direction,
nothing drifts, and there is no accumulated error to worry about.

The second is that almost nothing changes from one frame to the next.
Every transition a star will ever make is precomputed into one sorted
event schedule. Stepping forward is then a binary search plus a walk
over the handful of stars that actually crossed a threshold, rather
than a pass over the whole population. A full O(N) evaluation is kept
for jumps and for running the clock backwards.
"""

import numpy as np

from simulation_constants import (
    YEAR,
    MYR,
    GYR,
    PARSEC,
)

from stellar_pop.stellar_evolution.relations import (
    main_sequence_luminosity,
    main_sequence_radius,
    effective_temperature,
    main_sequence_lifetime,
    giant_phase_duration,
    giant_radius,
    giant_luminosity,
    stellar_fate,
    FATE_WHITE_DWARF,
    FATE_NEUTRON_STAR,
    FATE_BLACK_HOLE,
    FATE_DIRECT_COLLAPSE,
    FATE_PAIR_INSTABILITY,
)

from stellar_pop.stellar_evolution.blackbody import temperature_to_rgb

from stellar_pop.remnants import (
    remnant_mass,
    remnant_radius_solar,
    remnant_temperature,
    remnant_luminosity_solar,
)

from stellar_pop.supernovae import (
    classify as classify_supernova,
    nickel_mass,
    light_curve,
    SN_TYPE_NAMES,
)

from stellar_pop.star_birth import (
    sample_imf,
    sample_formation_times,
    sample_metallicity,
    formation_time_cdf,
    imf_pdf,
    log_uniform_pdf,
)

# Phase codes.
PHASE_UNBORN = 0
PHASE_MAIN_SEQUENCE = 1
PHASE_GIANT = 2
PHASE_REMNANT = 3

PHASE_NAMES = {
    PHASE_UNBORN: "not yet formed",
    PHASE_MAIN_SEQUENCE: "main sequence",
    PHASE_GIANT: "giant branch",
    PHASE_REMNANT: "remnant",
}

# How long a supernova stays visibly bright.
SUPERNOVA_VISIBLE_FOR = 3.0 * YEAR

# White dwarfs and neutron stars cool continuously but far too slowly
# to be worth recomputing every frame, and they are far too numerous
# to keep in the transient set. Instead the whole population is swept
# in a rolling window, so every star is brought up to date at least
# once per this many frames.
REFRESH_SWEEP_FRAMES = 60

# Luminosity floor, so log10 never sees a zero.
MIN_LUMINOSITY = 1.0e-12

# Layout of the render block: the six per-star floats the vertex
# buffer needs, kept contiguous so uploading them is one copy rather
# than four strided ones.
RENDER_RED = 0
RENDER_ALPHA = 3
RENDER_LOG_LUMINOSITY = 4
RENDER_SIZE_GAIN = 5
RENDER_WIDTH = 6


class _State:
    """
    The mutable part of the population that a refresh depends on.

    Bundling it lets a refresh be computed against a *proposed* state
    on a worker thread, rather than only against the live one.
    """

    __slots__ = (
        "phase",
        "giant_start",
        "giant_duration",
        "death_time",
        "forced_supernova_time",
        "explodes",
    )

    def __init__(self, phase, giant_start, giant_duration, death_time, forced, explodes):
        self.phase = phase
        self.giant_start = giant_start
        self.giant_duration = giant_duration
        self.death_time = death_time
        self.forced_supernova_time = forced
        self.explodes = explodes


class _Appearance:
    """
    Where a refresh writes its results.
    """

    __slots__ = ("luminosity", "temperature", "radius", "render_block")

    def __init__(self, count):
        self.luminosity = np.zeros(count, dtype=np.float32)
        self.temperature = np.zeros(count, dtype=np.float32)
        self.radius = np.zeros(count, dtype=np.float32)
        self.render_block = np.zeros((count, RENDER_WIDTH), dtype=np.float32)

# ------------------------------------------------------------
# Sampling the population
# ------------------------------------------------------------
#
# A real spiral galaxy holds about 1e11 stars, of which perhaps one in
# a million is an O or B star massive enough to end in a supernova.
# Draw a quarter of a million stars straight from the initial mass
# function and the expected number of living massive stars is *zero* -
# the population would be entirely red dwarfs and white dwarfs, with
# no blue stars, no giants worth looking at and nothing that could
# ever explode.
#
# That is not a flaw in the mass function; it is a sampling problem,
# and it has a standard answer. The population is drawn from a
# deliberate mixture:
#
#   * a field component from the true Kroupa IMF, with formation times
#     following the cosmic star formation history - the real
#     demographics, overwhelmingly old and low mass;
#
#   * a young component drawn uniformly in log mass over the range
#     that actually dominates a galaxy's light, with ages drawn
#     uniformly across each star's own lifetime. For a galaxy forming
#     stars at a steady rate that uniform age distribution is not an
#     approximation - it is the exact equilibrium distribution.
#
# Every star then carries an importance weight, the ratio of the true
# IMF density to the density it was actually drawn from, so weighted
# counts still reproduce the real population. What is over-sampled is
# exactly what a telescope over-samples: the rare, bright, short-lived
# stars that a galaxy's appearance is made of.

YOUNG_FRACTION = 0.35

YOUNG_MASS_MIN = 0.9
YOUNG_MASS_MAX = 130.0


class StellarPopulation:
    """
    Every star of one galaxy.
    """

    def __init__(self, count, rng, expansion, galaxy_radius_m, morphology, positions_m=None):
        self.count = int(count)
        self.expansion = expansion
        self.rng = rng

        # ---- Birth properties, fixed for the life of the star ----

        self.is_young = rng.random(self.count) < YOUNG_FRACTION

        n_young = int(self.is_young.sum())

        mass = sample_imf(self.count, rng)

        mass[self.is_young] = np.exp(
            rng.uniform(np.log(YOUNG_MASS_MIN), np.log(YOUNG_MASS_MAX), n_young)
        )

        self.mass = mass.astype(np.float32)

        self.weight = self._importance_weights(mass)

        self._formation_cdf, self._formation_times = formation_time_cdf(expansion)

        self.formation_time = sample_formation_times(
            self.count, rng, expansion, self._formation_cdf, self._formation_times
        )

        self.metallicity = sample_metallicity(
            self.formation_time, rng, expansion
        ).astype(np.float32)

        if positions_m is None:
            from galaxy_pop.galaxy_formation import galaxy_star_positions

            positions_m = galaxy_star_positions(
                self.count, rng, morphology, galaxy_radius_m
            )

        self.position_pc = (positions_m / PARSEC).astype(np.float32)

        # ---- Derived timescales ----

        mass64 = self.mass.astype(np.float64)

        self.ms_lifetime = main_sequence_lifetime(mass64, self.metallicity)

        self.giant_duration = giant_phase_duration(mass64, self.ms_lifetime)

        self.total_lifetime = self.ms_lifetime + self.giant_duration

        # The young component is redrawn so each star sits at a random
        # point in its own life at the reference epoch, which is the
        # steady-state age distribution for ongoing star formation.
        self._seed_young_ages(expansion.present_age)

        self.giant_start = self.formation_time + self.ms_lifetime
        self.death_time = self.giant_start + self.giant_duration

        # ---- Main-sequence structure ----

        ms_lum = main_sequence_luminosity(mass64)
        ms_rad = main_sequence_radius(mass64)

        self.ms_luminosity = ms_lum.astype(np.float32)
        self.ms_radius = ms_rad.astype(np.float32)
        self.ms_temperature = effective_temperature(ms_lum, ms_rad).astype(np.float32)

        # ---- Fate and remnant structure ----

        self.fate = stellar_fate(mass64)

        self.remnant_mass = remnant_mass(mass64, self.fate).astype(np.float32)
        self.remnant_radius = remnant_radius_solar(
            self.remnant_mass.astype(np.float64), self.fate
        ).astype(np.float32)

        self.sn_type = classify_supernova(mass64, self.metallicity)
        self.nickel_mass = nickel_mass(mass64, self.sn_type).astype(np.float32)

        # Stars below the core-collapse limit die quietly.
        self.explodes = (
            (self.fate == FATE_NEUTRON_STAR)
            | (self.fate == FATE_BLACK_HOLE)
            | (self.fate == FATE_PAIR_INSTABILITY)
        )

        # ---- Mutable render state ----

        self.phase = np.zeros(self.count, dtype=np.int8)

        self._bind_appearance(_Appearance(self.count))

        self.log_luminosity[:] = -12.0
        self.size_gain[:] = 1.0

        # Stars the user has manually detonated.
        self.forced_supernova_time = np.full(self.count, np.inf)

        # ---- Spatial index, for nearest-object queries ----

        self._build_spatial_index()

        self._nearest_cache = None

        # ---- Event schedule ----

        self._build_event_schedule()

        self.cosmic_time = -1.0
        self._event_cursor = 0
        self._sweep_cursor = 0
        self._sweep_size = max(1, self.count // REFRESH_SWEEP_FRAMES)

        self.evaluate_all(expansion.present_age)

    # --------------------------------------------------------
    # Render block
    # --------------------------------------------------------

    def _bind_appearance(self, appearance):
        """
        Adopt a set of appearance arrays, exposing the render block's
        columns as named views.
        """

        self.luminosity = appearance.luminosity
        self.temperature = appearance.temperature
        self.radius = appearance.radius

        block = appearance.render_block

        self.render_block = block

        self.colour = block[:, RENDER_RED : RENDER_RED + 3]
        self.alpha = block[:, RENDER_ALPHA]
        self.log_luminosity = block[:, RENDER_LOG_LUMINOSITY]
        self.size_gain = block[:, RENDER_SIZE_GAIN]

    def _live_state(self):
        return _State(
            self.phase,
            self.giant_start,
            self.giant_duration,
            self.death_time,
            self.forced_supernova_time,
            self.explodes,
        )

    def _live_appearance(self):
        appearance = _Appearance.__new__(_Appearance)

        appearance.luminosity = self.luminosity
        appearance.temperature = self.temperature
        appearance.radius = self.radius
        appearance.render_block = self.render_block

        return appearance

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    def _importance_weights(self, mass):
        """
        How many real stars each simulated star stands for, up to an
        overall normalisation.

        w = p_true(m) / q(m), where q is the mixture the sample was
        actually drawn from. Weighted sums over the population are then
        unbiased estimates of the real galaxy.
        """

        true_density = imf_pdf(mass)

        young_density = log_uniform_pdf(mass, YOUNG_MASS_MIN, YOUNG_MASS_MAX)

        mixture = (
            1.0 - YOUNG_FRACTION
        ) * true_density + YOUNG_FRACTION * young_density

        return (true_density / np.maximum(mixture, 1.0e-300)).astype(np.float64)

    def _seed_young_ages(self, reference_time):
        """
        Give the young component a steady-state age distribution about
        a reference epoch: each star's age is uniform on [0, its own
        total lifetime], truncated so nothing is born before its
        galaxy.

        Called again whenever the clock moves far enough that the young
        population should have turned over, which is what ongoing star
        formation actually does.
        """

        young = self.is_young

        if not young.any():
            return

        lifetimes = np.minimum(self.total_lifetime[young], 8.0 * GYR)

        ages = self.rng.random(int(young.sum())) * lifetimes

        self.formation_time[young] = np.maximum(reference_time - ages, 0.18 * GYR)

        self._young_reference_time = reference_time

    def reseed_young_population(self, cosmic_time):
        """
        Refresh the young population for a new epoch and rebuild the
        derived timelines.

        A galaxy that is still forming stars does not keep the same
        O stars for a billion years, so after a large time jump the
        short-lived component has to be drawn again.
        """

        return self.apply_young_reseed(
            self.plan_young_reseed(cosmic_time, self.rng), cosmic_time
        )

    def plan_young_reseed(self, cosmic_time, rng):
        """
        Work out the new young population without touching any state.

        Everything here only reads, so it is safe to run on a worker
        thread while the main thread keeps drawing. Re-sorting the
        event schedule is the expensive part, and doing it in the
        middle of a frame is exactly the kind of stall that shows up
        as a stutter.
        """

        young = self.is_young

        formation_time = self.formation_time.copy()

        if young.any():
            lifetimes = np.minimum(self.total_lifetime[young], 8.0 * GYR)

            ages = rng.random(int(young.sum())) * lifetimes

            formation_time[young] = np.maximum(cosmic_time - ages, 0.18 * GYR)

        giant_start = formation_time + self.ms_lifetime
        death_time = giant_start + self.giant_duration

        times = np.concatenate([formation_time, giant_start, death_time])

        order = np.argsort(times, kind="stable")

        # Everything below is what evaluate_all would otherwise do on
        # the main thread: several tens of milliseconds, which is four
        # dropped frames if it lands inside one.
        phase = (
            (cosmic_time >= formation_time).astype(np.int8)
            + (cosmic_time >= giant_start)
            + (cosmic_time >= death_time)
        )

        forced = np.full(self.count, np.inf)

        state = _State(
            phase,
            giant_start,
            self.giant_duration,
            death_time,
            forced,
            self.explodes,
        )

        appearance = _Appearance(self.count)

        everything = np.arange(self.count)

        self._refresh(everything, cosmic_time, state=state, out=appearance)

        transient = np.flatnonzero(
            self._transient_mask(everything, cosmic_time, state=state)
        )

        return {
            "formation_time": formation_time,
            "giant_start": giant_start,
            "death_time": death_time,
            "event_time": times[order],
            "event_index": self._event_index_template[order],
            "phase": phase,
            "forced_supernova_time": forced,
            "appearance": appearance,
            "transient": transient,
            "reference_time": cosmic_time,
        }

    def apply_young_reseed(self, plan, cosmic_time):
        """
        Install a plan produced by plan_young_reseed.

        Main thread only, and deliberately nothing but pointer swaps:
        all the arithmetic already happened on the worker.
        """

        self.formation_time = plan["formation_time"]
        self.giant_start = plan["giant_start"]
        self.death_time = plan["death_time"]

        self.event_time = plan["event_time"]
        self.event_index = plan["event_index"]

        self.phase = plan["phase"]
        self.forced_supernova_time = plan["forced_supernova_time"]

        self._bind_appearance(plan["appearance"])

        self._transient_indices = plan["transient"]

        self._young_reference_time = plan["reference_time"]

        self._nearest_cache = None

        self.cosmic_time = plan["reference_time"]

        self._event_cursor = int(
            np.searchsorted(self.event_time, self.cosmic_time, "right")
        )
        self._sweep_cursor = 0

        # If the clock moved on while the plan was being built, catch
        # up the difference rather than throwing the work away.
        if cosmic_time != self.cosmic_time:
            return self.advance_to(cosmic_time)

        return np.arange(self.count)

    def young_population_is_stale(self, cosmic_time, tolerance=200.0 * MYR):
        return abs(cosmic_time - self._young_reference_time) > tolerance

    # --------------------------------------------------------
    # Spatial index
    # --------------------------------------------------------

    def _build_spatial_index(self, resolution=256):
        """
        A uniform grid over the galaxy, stored as stars sorted by cell.

        Star positions never change, so this is built once. Finding the
        nearest star then means looking in a handful of cells rather
        than measuring the distance to all quarter of a million, which
        was costing more per frame than the entire physics update.
        """

        positions = self.position_pc

        self._grid_low = positions.min(axis=0).astype(np.float64)

        span = np.maximum(
            positions.max(axis=0).astype(np.float64) - self._grid_low, 1.0e-3
        )

        self._grid_resolution = int(resolution)

        # Cubic cells. A galaxy disc is a hundred times wider than it
        # is thick, so scaling each axis independently would give cells
        # that are long slabs, and a search radius measured in cells
        # would mean wildly different distances along each axis.
        self._grid_cell = float(span.max()) / self._grid_resolution

        self._grid_high = self._grid_low + span

        cells = self._cell_of(positions)

        key = self._cell_key(cells)

        order = np.argsort(key, kind="stable")

        self._grid_order = order.astype(np.int32)
        self._grid_keys = key[order]

    def _cell_of(self, positions, clamp=True):
        cells = (
            np.asarray(positions, dtype=np.float64) - self._grid_low
        ) / self._grid_cell

        cells = np.floor(cells).astype(np.int64)

        if clamp:
            cells = np.clip(cells, 0, self._grid_resolution - 1)

        return cells

    def _cell_key(self, cells):
        r = self._grid_resolution

        cells = np.atleast_2d(cells)

        return (cells[:, 0] * r + cells[:, 1]) * r + cells[:, 2]

    def _candidates_near(self, position_pc, ring):
        """
        Indices of stars in the cells within `ring` cells of a point.
        """

        centre = self._cell_of(np.atleast_2d(position_pc))[0]

        offsets = np.arange(-ring, ring + 1)

        dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")

        neighbours = np.stack(
            [
                np.clip(centre[0] + dx.ravel(), 0, self._grid_resolution - 1),
                np.clip(centre[1] + dy.ravel(), 0, self._grid_resolution - 1),
                np.clip(centre[2] + dz.ravel(), 0, self._grid_resolution - 1),
            ],
            axis=1,
        )

        keys = np.unique(self._cell_key(neighbours))

        starts = np.searchsorted(self._grid_keys, keys, "left")
        ends = np.searchsorted(self._grid_keys, keys, "right")

        spans = [
            self._grid_order[start:end]
            for start, end in zip(starts, ends)
            if end > start
        ]

        if not spans:
            return None

        return np.concatenate(spans)

    # --------------------------------------------------------
    # Event schedule
    # --------------------------------------------------------

    def _build_event_schedule(self):
        """
        Every phase transition any star will ever make, sorted by time.

        Three per star: formation, the end of the main sequence, and
        death. Stepping the clock forward then costs a binary search
        plus one touch per star that actually changed.
        """

        self._event_index_template = np.tile(
            np.arange(self.count, dtype=np.int32), 3
        )

        times = np.concatenate(
            [self.formation_time, self.giant_start, self.death_time]
        )

        order = np.argsort(times, kind="stable")

        self.event_time = times[order]
        self.event_index = self._event_index_template[order]

    # --------------------------------------------------------
    # Phase evaluation
    # --------------------------------------------------------

    def _phase_at(self, cosmic_time, indices=None):
        """
        Phase code for the given stars (or all of them) at a time.

        The three thresholds are ordered, so summing three comparisons
        is both the cheapest and the clearest way to get the code.
        """

        if indices is None:
            formation = self.formation_time
            giant_start = self.giant_start
            death = self.death_time
        else:
            formation = self.formation_time[indices]
            giant_start = self.giant_start[indices]
            death = self.death_time[indices]

        return (
            (cosmic_time >= formation).astype(np.int8)
            + (cosmic_time >= giant_start)
            + (cosmic_time >= death)
        )

    def evaluate_all(self, cosmic_time):
        """
        Recompute every star from scratch.

        Used at startup, after a time jump, and whenever the clock runs
        backwards. O(N), a few milliseconds for a quarter of a million
        stars, and exact regardless of how far the clock moved.
        """

        self.phase = self._phase_at(cosmic_time)

        everything = np.arange(self.count)

        self._refresh(everything, cosmic_time)

        self.cosmic_time = cosmic_time

        self._event_cursor = int(np.searchsorted(self.event_time, cosmic_time, "right"))
        self._sweep_cursor = 0

        self._transient_indices = np.flatnonzero(
            self._transient_mask(everything, cosmic_time)
        )

        return everything

    def advance_to(self, cosmic_time):
        """
        Move the clock and return the indices whose render state
        changed.

        Forward motion walks the event schedule. Anything else falls
        back to a full evaluation.
        """

        if cosmic_time < self.cosmic_time:
            return self.evaluate_all(cosmic_time)

        end = int(np.searchsorted(self.event_time, cosmic_time, "right"))

        changed = self.event_index[self._event_cursor : end]

        self._event_cursor = end
        self.cosmic_time = cosmic_time

        # The stars that need attention this frame are those that just
        # crossed a threshold plus those already changing continuously:
        # giants climbing the giant branch and supernovae fading. Both
        # sets are small, so nothing here scales with the population.
        if changed.size:
            candidates = np.unique(
                np.concatenate([changed, self._transient_indices])
            )
        else:
            candidates = self._transient_indices

        if candidates.size:
            # A star can cross two thresholds in one step, so recompute
            # the phase rather than trusting the event's own label.
            self.phase[candidates] = self._phase_at(cosmic_time, candidates)

        # Cooling remnants change over billions of years: far too slow
        # to keep in the transient set, far too numerous to ignore. A
        # rolling slice of the population is brought up to date each
        # frame instead, so every star is current to within a second of
        # wall clock at any time rate.
        swept = self._next_sweep_slice()

        dirty = (
            np.concatenate([candidates, swept]) if candidates.size else swept
        )

        # One refresh over everything dirty rather than two passes: the
        # per-call overhead of the vectorised relations dominates at
        # these array sizes.
        self._refresh(dirty, cosmic_time)

        if candidates.size:
            self._transient_indices = candidates[
                self._transient_mask(candidates, cosmic_time)
            ]

        return dirty

    def _next_sweep_slice(self):
        start = self._sweep_cursor
        end = min(start + self._sweep_size, self.count)

        self._sweep_cursor = 0 if end >= self.count else end

        return np.arange(start, end)

    def _transient_mask(self, indices, cosmic_time, state=None):
        """
        Which of the given stars are still changing from frame to
        frame: giants climbing the giant branch, and supernovae fading.
        """

        if state is None:
            state = self._live_state()

        phase = state.phase[indices]

        giant = phase == PHASE_GIANT

        exploding = (
            state.explodes[indices]
            & (phase == PHASE_REMNANT)
            & ((cosmic_time - state.death_time[indices]) < SUPERNOVA_VISIBLE_FOR)
        )

        forced_age = cosmic_time - state.forced_supernova_time[indices]
        forced = (forced_age >= 0.0) & (forced_age < SUPERNOVA_VISIBLE_FOR)

        return giant | exploding | forced

    # --------------------------------------------------------
    # Appearance
    # --------------------------------------------------------

    def _refresh(self, indices, cosmic_time, state=None, out=None):
        """
        Recompute luminosity, temperature, radius and render colour for
        the given stars.

        `state` and `out` default to this population's live arrays. A
        worker thread can pass a proposed state and somewhere else to
        write, which is how a whole new population can be prepared
        without the main thread seeing a half-finished one.
        """

        if len(indices) == 0:
            return

        if state is None:
            state = self._live_state()

        if out is None:
            out = self._live_appearance()

        phase = state.phase[indices]
        mass = self.mass[indices].astype(np.float64)

        luminosity = np.zeros(len(indices))
        temperature = np.zeros(len(indices))
        radius = np.zeros(len(indices))

        # --- Main sequence ---
        on_ms = phase == PHASE_MAIN_SEQUENCE

        if on_ms.any():
            rows = indices[on_ms]
            luminosity[on_ms] = self.ms_luminosity[rows]
            temperature[on_ms] = self.ms_temperature[rows]
            radius[on_ms] = self.ms_radius[rows]

        # --- Giant branch ---
        is_giant = phase == PHASE_GIANT

        if is_giant.any():
            rows = indices[is_giant]

            fraction = np.clip(
                (cosmic_time - state.giant_start[rows])
                / np.maximum(state.giant_duration[rows], 1.0),
                0.0,
                1.0,
            )

            giant_mass = self.mass[rows].astype(np.float64)

            lum = giant_luminosity(giant_mass, fraction)
            rad = giant_radius(giant_mass, fraction)

            luminosity[is_giant] = lum
            radius[is_giant] = rad
            temperature[is_giant] = effective_temperature(lum, rad)

        # --- Remnants, including the supernova flash ---
        dead = phase == PHASE_REMNANT

        if dead.any():
            rows = indices[dead]

            age = np.maximum(cosmic_time - state.death_time[rows], 0.0)

            temperature[dead] = remnant_temperature(
                age, self.remnant_mass[rows].astype(np.float64), self.fate[rows]
            )

            radius[dead] = self.remnant_radius[rows]

            luminosity[dead] = remnant_luminosity_solar(
                radius[dead], temperature[dead]
            )

            # Supernova light curve, added on top of the remnant.
            explodes = state.explodes[rows]

            if explodes.any():
                flash = light_curve(
                    age, self.nickel_mass[rows].astype(np.float64), self.sn_type[rows]
                )

                visible = explodes & (age < SUPERNOVA_VISIBLE_FOR)

                luminosity[dead] = np.where(
                    visible, luminosity[dead] + flash, luminosity[dead]
                )

                # A supernova photosphere is hot and blue-white.
                temperature[dead] = np.where(
                    visible & (age < 0.1 * YEAR), 12_000.0, temperature[dead]
                )

        # --- Manual supernovae ---
        forced_age = cosmic_time - state.forced_supernova_time[indices]
        forced = (forced_age >= 0.0) & (forced_age < SUPERNOVA_VISIBLE_FOR)

        if forced.any():
            rows = indices[forced]

            flash = light_curve(
                forced_age[forced],
                self.nickel_mass[rows].astype(np.float64),
                self.sn_type[rows],
            )

            luminosity[forced] = luminosity[forced] + flash
            temperature[forced] = np.maximum(temperature[forced], 12_000.0)

        # --- Write back ---
        out.luminosity[indices] = luminosity
        out.temperature[indices] = temperature
        out.radius[indices] = radius

        # Unborn stars and black holes emit nothing.
        visible = (phase != PHASE_UNBORN) & (luminosity > MIN_LUMINOSITY)

        # Giants and supernovae are physically much larger, so they get
        # a wider point spread as well as a brighter one.
        gain = np.where(
            is_giant, 1.6, np.where(dead & state.explodes[indices], 3.0, 1.0)
        )

        # One contiguous write, so the renderer can upload these rows
        # with a single copy.
        block = np.empty((len(indices), RENDER_WIDTH), dtype=np.float32)

        block[:, RENDER_RED : RENDER_RED + 3] = temperature_to_rgb(
            np.maximum(temperature, 500.0)
        )
        block[:, RENDER_ALPHA] = visible
        block[:, RENDER_LOG_LUMINOSITY] = np.log10(
            np.maximum(luminosity, MIN_LUMINOSITY)
        )
        block[:, RENDER_SIZE_GAIN] = gain

        out.render_block[indices] = block

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    def force_supernova(self, index, cosmic_time):
        """
        Detonate a star on demand.

        Only a star that is still burning can be made to explode, and
        only one massive enough to reach core collapse. A red dwarf
        cannot be talked into it.
        """

        index = int(index)

        if not (0 <= index < self.count):
            return None

        if self.phase[index] not in (PHASE_MAIN_SEQUENCE, PHASE_GIANT):
            return None

        if self.mass[index] < 8.0:
            return None

        # Bring the star's death forward to now.
        self.forced_supernova_time[index] = cosmic_time

        self.death_time[index] = cosmic_time
        self.giant_start[index] = min(self.giant_start[index], cosmic_time)

        self.phase[index] = PHASE_REMNANT

        self.explodes[index] = True

        rows = np.array([index])

        self._refresh(rows, cosmic_time)

        self._transient_indices = np.unique(
            np.concatenate([self._transient_indices, rows])
        )

        return SN_TYPE_NAMES[int(self.sn_type[index])]

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    def counts_by_phase(self, weighted=False):
        """
        Population counts. Weighted counts estimate the real galaxy;
        unweighted counts describe what is on screen.
        """

        weights = self.weight if weighted else None

        def total(mask):
            return float(weights[mask].sum()) if weighted else int(mask.sum())

        return {
            PHASE_NAMES[code]: total(self.phase == code)
            for code in (PHASE_UNBORN, PHASE_MAIN_SEQUENCE, PHASE_GIANT, PHASE_REMNANT)
        }

    def remnant_counts(self, weighted=False):
        dead = self.phase == PHASE_REMNANT

        weights = self.weight if weighted else None

        def total(mask):
            return float(weights[mask].sum()) if weighted else int(mask.sum())

        return {
            "white dwarfs": total(dead & (self.fate == FATE_WHITE_DWARF)),
            "neutron stars": total(dead & (self.fate == FATE_NEUTRON_STAR)),
            "black holes": total(
                dead
                & (
                    (self.fate == FATE_BLACK_HOLE)
                    | (self.fate == FATE_DIRECT_COLLAPSE)
                )
            ),
        }

    def massive_living_stars(self, minimum_mass=8.0):
        """
        Indices of stars still burning that could be made to explode.
        """

        alive = (self.phase == PHASE_MAIN_SEQUENCE) | (self.phase == PHASE_GIANT)

        return np.flatnonzero(alive & (self.mass >= minimum_mass))

    def active_supernovae(self, cosmic_time):
        """
        Indices of stars currently exploding.

        Anything that is exploding is by definition changing from
        frame to frame, so it is already in the transient set. Scanning
        that handful of indices costs nothing; scanning the whole
        population every frame cost more than the physics did.
        """

        candidates = self._transient_indices

        if candidates.size == 0:
            return candidates

        age = cosmic_time - self.death_time[candidates]

        natural = (
            self.explodes[candidates] & (age >= 0.0) & (age < SUPERNOVA_VISIBLE_FOR)
        )

        forced_age = cosmic_time - self.forced_supernova_time[candidates]
        forced = (forced_age >= 0.0) & (forced_age < SUPERNOVA_VISIBLE_FOR)

        return candidates[natural | forced]

    def nearest(self, position_pc, tolerance=0.25):
        """
        Index of the star closest to a point, in parsecs.

        Answered from the spatial grid, and cached: the result is
        reused until the observer has moved by an appreciable fraction
        of the distance to it. Far from the galaxy that means the
        answer is computed once and then held for thousands of frames;
        close in, the grid makes recomputing it cheap anyway.
        """

        target = np.asarray(position_pc, dtype=np.float64)

        cached = self._nearest_cache

        if cached is not None:
            index, origin, distance = cached

            if np.linalg.norm(target - origin) < tolerance * distance:
                return index

        index, distance = self._nearest_uncached(target)

        self._nearest_cache = (index, target.copy(), max(distance, 1.0e-6))

        return index

    # The grid is only searched a few cells out. Beyond that the
    # neighbourhood is larger than a brute-force scan is expensive, so
    # the scan is the better answer.
    MAX_SEARCH_RING = 3

    def _nearest_uncached(self, target):
        """
        Search outward through the grid, then fall back.

        A grid answer is only accepted once the best distance found is
        inside the box already searched; otherwise a nearer star could
        still be hiding one ring further out.
        """

        raw_cell = self._cell_of(np.atleast_2d(target), clamp=False)[0]

        inside = np.all(
            (raw_cell >= -self.MAX_SEARCH_RING)
            & (raw_cell < self._grid_resolution + self.MAX_SEARCH_RING)
        )

        if inside:
            for ring in range(1, self.MAX_SEARCH_RING + 1):
                candidates = self._candidates_near(target, ring)

                if candidates is None:
                    continue

                delta = self.position_pc[candidates] - target.astype(np.float32)

                distances = np.einsum("ij,ij->i", delta, delta)

                best = int(np.argmin(distances))
                distance = float(np.sqrt(distances[best]))

                if distance <= ring * self._grid_cell:
                    return int(candidates[best]), distance

        return self._nearest_by_scan(target)

    def _nearest_by_scan(self, target):
        """
        Exact nearest star by brute force.

        Used when the observer is outside the galaxy, where the cache
        then holds the answer for a very long time, because the
        tolerance scales with how far away the star is.
        """

        delta = self.position_pc - np.asarray(target, dtype=np.float32)

        distances = np.einsum("ij,ij->i", delta, delta)

        best = int(np.argmin(distances))

        return best, float(np.sqrt(distances[best]))

    def describe(self, index):
        """
        A human-readable summary of one star.
        """

        index = int(index)

        from stellar_pop.stellar_evolution.blackbody import spectral_class
        from stellar_pop.stellar_evolution.relations import FATE_NAMES

        phase = int(self.phase[index])

        lines = {
            "mass": f"{self.mass[index]:.3f} Msun",
            "phase": PHASE_NAMES[phase],
            "luminosity": f"{self.luminosity[index]:.4g} Lsun",
            "radius": f"{self.radius[index]:.4g} Rsun",
            "temperature": f"{self.temperature[index]:,.0f} K",
            "class": spectral_class(self.temperature[index]),
            "metallicity": f"{self.metallicity[index]:.5f}",
            "born": f"{self.formation_time[index] / GYR:.3f} Gyr",
            "main-sequence life": f"{self.ms_lifetime[index] / MYR:,.1f} Myr",
            "fate": FATE_NAMES[int(self.fate[index])],
        }

        if phase == PHASE_REMNANT:
            lines["remnant mass"] = f"{self.remnant_mass[index]:.3f} Msun"

        return lines
