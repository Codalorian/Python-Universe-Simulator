"""
Distant galaxies.

Sixty thousand galaxies spread across ten gigaparsecs, drawn as one
point batch. Each is a soft glow whose colour comes from its stellar
population and whose brightness comes from its stellar mass and its
distance, so the cosmic web appears the way a deep survey shows it:
bright nearby spirals, faint red smudges at high redshift.

Galaxy positions are comoving, and the vertex shader multiplies them
by a(t). Expansion therefore costs one uniform, and running the clock
backwards genuinely contracts the universe.
"""

import numpy as np

import numpy as np

from simulation_constants import MPC, C, YEAR, SUN_LUMINOSITY

from renderer import PointCloud, LIGHT_YEAR

from galaxy_pop.galaxy_formation import (
    MORPH_SPIRAL,
    MORPH_ELLIPTICAL,
    MORPH_IRREGULAR,
    MORPH_LENTICULAR,
    MORPH_DWARF,
)

MPC_IN_LIGHT_YEARS = MPC / LIGHT_YEAR

# Rest-frame colours by morphology. Ellipticals are red and dead;
# spirals and irregulars are blue with ongoing star formation.
MORPHOLOGY_COLOURS = {
    MORPH_SPIRAL: (0.82, 0.85, 1.00),
    MORPH_ELLIPTICAL: (1.00, 0.84, 0.66),
    MORPH_IRREGULAR: (0.70, 0.84, 1.00),
    MORPH_LENTICULAR: (1.00, 0.91, 0.80),
    MORPH_DWARF: (0.86, 0.88, 0.96),
}


class GalaxyField:
    def __init__(self, catalogue, expansion, parent=None, shader=None):
        self.catalogue = catalogue
        self.expansion = expansion

        count = len(catalogue["halo_mass"])

        self.cloud = PointCloud(
            count=count,
            unit_in_light_years=MPC_IN_LIGHT_YEARS,
            parent=parent,
            shader=shader,
            size_gain_pixels=3.4,
            max_point_size=150.0,
            min_point_size=0.0,
            faint_cutoff=-12.5,
            brightness=1.0,
        )

        self.cloud.set_positions(catalogue["position_mpc"])

        self.base_colour = self._morphology_colours()

        # A galaxy's luminosity is roughly its stellar mass divided by
        # a mass-to-light ratio of a few, so this is log10(L / Lsun).
        self.base_log_luminosity = np.log10(
            np.maximum(catalogue["stellar_mass"], 1.0) / 3.0
        ).astype(np.float32)

        # Bigger galaxies are resolved as larger smudges rather than
        # just brighter points.
        self.size_gain = np.clip(
            (catalogue["radius"] / (3.0 * 3.0857e19)) ** 0.35, 0.5, 3.0
        ).astype(np.float32)

        self.cloud.data[:, 8] = self.size_gain

        # A galaxy's own colour and brightness are fixed by what it is
        # and where it is, not by when you look, so they are computed
        # once here.
        self.tinted_colour = (self.base_colour * self._redshift_tint()).astype(
            np.float32
        )

        # The block holds the six per-galaxy floats the vertex buffer
        # wants, contiguous, so an update is one copy.
        self.block = np.zeros((count, 6), dtype=np.float32)

        self.block[:, 0:3] = self.tinted_colour
        self.block[:, 4] = self.base_log_luminosity
        self.block[:, 5] = self.size_gain

        self.scale_factor = 1.0

        self._last_quasars = False

        # The galaxy the observer is inside is drawn star by star, so
        # its own smudge in the catalogue would be a blob centred on
        # the viewer.
        self.host_index = None
        self._last_time = self.expansion.present_age

        self.update(self.expansion.present_age)

    # --------------------------------------------------------

    def _morphology_colours(self):
        morphology = self.catalogue["morphology"]

        colours = np.zeros((len(morphology), 3), dtype=np.float32)

        for code, rgb in MORPHOLOGY_COLOURS.items():
            colours[morphology == code] = rgb

        return colours

    def update(self, cosmic_time, quasar_luminosity_watts=None):
        """
        Recompute what is visible at this cosmic time.

        Galaxies that have not formed yet are invisible; those that
        have brighten in as they assemble. A galaxy whose central black
        hole is currently accreting is far brighter than its stars and
        is drawn as the blue-white point a quasar actually is.
        """

        catalogue = self.catalogue

        formed = catalogue["formation_time"] <= cosmic_time
        alive = catalogue["halo_mass"] > 0.0

        visible = formed & alive

        # How long the galaxy has been assembling, as a fraction of a
        # typical assembly time: newly formed galaxies brighten in.
        age = np.maximum(cosmic_time - catalogue["formation_time"], 0.0)
        ramp = np.clip(age / (3.0e8 * 3.156e7), 0.0, 1.0)

        self.block[:, 3] = np.where(visible, ramp, 0.0)

        if self.host_index is not None:
            self.block[self.host_index, 3] = 0.0

        # _update_quasars builds a fresh array whenever the active
        # population changes, so identity is an exact and free test for
        # "did anything but visibility change".
        quasars_changed = quasar_luminosity_watts is not self._last_quasars

        self._last_quasars = quasar_luminosity_watts

        if quasar_luminosity_watts is None:
            self.block[:, 0:3] = self.tinted_colour
            self.block[:, 4] = self.base_log_luminosity
        else:
            quasar_solar = np.asarray(quasar_luminosity_watts) / SUN_LUMINOSITY

            total = 10.0**self.base_log_luminosity + quasar_solar

            self.block[:, 4] = np.log10(np.maximum(total, 1.0))

            # An accretion disc peaks in the ultraviolet, so an active
            # nucleus outshines its host and turns it blue-white.
            active = quasar_solar > 10.0**self.base_log_luminosity

            self.block[:, 0:3] = np.where(
                active[:, None], np.float32([0.78, 0.86, 1.0]), self.tinted_colour
            )

            self.block[:, 5] = np.where(active, self.size_gain * 1.8, self.size_gain)

        if quasars_changed:
            self.cloud.data[:, 3:9] = self.block
            self.cloud.upload()
        else:
            # Only visibility moved, so write one column instead of
            # two megabytes.
            self.cloud.data[:, 6] = self.block[:, 3]
            self.cloud.upload_column(6)

        self._last_time = cosmic_time

        self.scale_factor = float(self.expansion.scale_factor(cosmic_time))

    def _redshift_tint(self):
        """
        A per-galaxy warm tint standing in for cosmological reddening.

        Full spectral redshifting of a galaxy template is beyond what a
        real-time renderer can do per object, but shifting the colour
        balance towards red with distance reproduces the look of a
        deep field, where the most distant galaxies are visibly the
        reddest things in the frame.
        """

        distance_mpc = np.linalg.norm(self.catalogue["position_mpc"], axis=1)

        # Approximate redshift from comoving distance.
        z = np.clip(distance_mpc / 3_500.0, 0.0, 8.0)

        warm = np.empty((len(z), 3), dtype=np.float32)
        warm[:, 0] = 1.0
        warm[:, 1] = 1.0 / (1.0 + 0.16 * z)
        warm[:, 2] = 1.0 / (1.0 + 0.38 * z)

        return warm

    # Reference brightness at the present day.
    BASE_BRIGHTNESS = 1.0

    def set_host(self, index):
        """
        Mark the galaxy whose stars are being drawn individually.
        """

        if index == self.host_index:
            return

        self.host_index = None if index is None else int(index)

        self.update(self.expansion.present_age if False else self._last_time)

    def set_camera_mpc(self, position_mpc, cosmic_time=None):
        scale = (
            self.scale_factor
            if cosmic_time is None
            else float(self.expansion.scale_factor(cosmic_time))
        )

        # Exposure compensation. Comoving separations shrink with a(t),
        # so at cosmic noon every galaxy is three and a half times
        # closer and thirteen times brighter, and the sky saturates to
        # white. Scaling the gain by a^2 is the same thing a camera
        # does when it stops down: the sky stays readable while the
        # genuine change - far more galaxies per unit sky - is still
        # plain to see.
        self.cloud.set_brightness(
            self.BASE_BRIGHTNESS * min(max(scale * scale, 0.02), 1.0)
        )

        self.cloud.set_camera(position_mpc, scale_factor=scale)

    def set_visible(self, value):
        self.cloud.visible = value

    def destroy(self):
        self.cloud.destroy()
