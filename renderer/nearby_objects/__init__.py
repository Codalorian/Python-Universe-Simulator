"""
Near-field rendering.

Everything the observer can resolve as more than a point: the star
they are nearest, its planets, and any black hole close enough to
show a shadow.

There are never more than a few dozen of these, so they can afford to
be real scene graph nodes with real geometry. Every pool is fixed in
size and reused, so flying between star systems allocates nothing.
"""

import numpy as np

import panda3d.core as p3d

from ursina import Entity, Vec2, Vec3, Vec4, color, scene
from ursina.shaders import unlit_shader

from simulation_constants import (
    SUN_RADIUS,
    SUN_TEMPERATURE,
    SUN_LUMINOSITY,
)

from renderer import compress_offset, compress_distance, LIGHT_YEAR
from renderer.geometry import (
    sphere_model,
    billboard_model,
    ring_outline,
)
from renderer.shaders import star_shader, planet_shader, corona_shader
from renderer.black_holes import BlackHoleVisual

from stellar_pop.stellar_evolution.blackbody import temperature_to_rgb

from planetary_systems.formation import PLANET_COLOURS

# Vertices per orbit ring.
RING_SEGMENTS = 128

_UNIT_CIRCLE = np.stack(
    [
        np.cos(np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS)),
        np.sin(np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS)),
        np.zeros(RING_SEGMENTS),
    ],
    axis=1,
)

# How many resolvable bodies can be on screen at once.
POOL_SIZE = 32
RING_COUNT = 12
BLACK_HOLE_COUNT = 3

# Inside this distance from the camera, a body is drawn as geometry.
DETAIL_RANGE_METRES = 0.5 * LIGHT_YEAR

# Smallest angular radius, in radians, at which a body is still drawn.
#
# To scale, an Earth-sized planet seen from across its own system is
# a fraction of a pixel, and the simulation would be telling the truth
# by drawing nothing. A floor of about three pixels keeps the worlds
# findable; everything larger is drawn at its true angular size, and
# the readout always gives the real numbers.
MIN_ANGULAR_RADIUS = 0.0038

# Limb darkening coefficient for a stellar photosphere in visible
# light: I(mu)/I(0) = 1 - u (1 - mu).
LIMB_DARKENING = 0.62

# Flux, in W/m^2, at which a source starts to throw visible glare.
# Roughly a naked-eye star at the limit of detection.
GLARE_THRESHOLD_FLUX = 1.0e-8

# Angular radius of glare added per decade of flux above that.
GLARE_PER_DECADE = 0.0042


class NearbyObjects:
    """
    A reusable pool of resolvable bodies.
    """

    def __init__(self, parent=None, overlay_discs=False):
        self.parent = parent or scene

        self.pool = []

        for _ in range(POOL_SIZE):
            entity = Entity(
                model=sphere_model(),
                parent=self.parent,
                scale=0.0,
                enabled=False,
                double_sided=False,
            )

            entity.setLightOff()
            entity.setTransparency(p3d.TransparencyAttrib.M_none)
            entity.model.setTransparency(p3d.TransparencyAttrib.M_none)

            # The compressed depth range spans seven orders of
            # magnitude, which no depth buffer can resolve, so these
            # are drawn back to front instead.
            entity.setDepthWrite(False)
            entity.setDepthTest(False)

            self.pool.append(entity)

        self.used = 0
        self._depth = [0.0] * POOL_SIZE
        self._bin_order = [-1] * POOL_SIZE

        # Glare billboards, drawn after everything solid.
        self.coronae = []

        for _ in range(8):
            glare = Entity(
                model=billboard_model(),
                parent=self.parent,
                enabled=False,
            )

            glare.shader = corona_shader()
            glare.setLightOff()
            glare.setTransparency(p3d.TransparencyAttrib.M_none)
            glare.model.setTransparency(p3d.TransparencyAttrib.M_none)
            glare.setDepthWrite(False)
            glare.setDepthTest(False)
            glare.setTwoSided(True)
            glare.setBin("fixed", 60)
            glare.setAttrib(
                p3d.ColorBlendAttrib.make(
                    p3d.ColorBlendAttrib.M_add,
                    p3d.ColorBlendAttrib.O_one,
                    p3d.ColorBlendAttrib.O_one,
                )
            )

            self.coronae.append(glare)

        self.coronae_used = 0

        # Orbit rings. Each keeps a permanent NumPy view of its own
        # vertex buffer, for the same reason the point clouds do.
        self.rings = []
        self.ring_buffers = []

        for _ in range(RING_COUNT):
            mesh = ring_outline(RING_SEGMENTS)

            ring = Entity(
                model=mesh,
                parent=self.parent,
                enabled=False,
                color=color.rgba32(120, 150, 210, 70),
            )

            ring.shader = unlit_shader
            ring.setLightOff()
            ring.setDepthWrite(False)
            ring.setDepthTest(False)
            ring.setBin("fixed", 15)
            ring.setAttrib(
                p3d.ColorBlendAttrib.make(
                    p3d.ColorBlendAttrib.M_add,
                    p3d.ColorBlendAttrib.O_incoming_alpha,
                    p3d.ColorBlendAttrib.O_one,
                )
            )

            self.rings.append(ring)
            self.ring_buffers.append(
                np.frombuffer(
                    memoryview(mesh.geom.modifyVertexData().modifyArray(0))
                    .cast("B")
                    .cast("f"),
                    dtype=np.float32,
                ).reshape(RING_SEGMENTS, 3)
            )

        self.rings_used = 0

        self.black_holes = [
            BlackHoleVisual(self.parent, overlay=overlay_discs)
            for _ in range(BLACK_HOLE_COUNT)
        ]
        self.black_holes_used = 0

    # --------------------------------------------------------
    # Frame boundaries
    # --------------------------------------------------------

    def begin_frame(self):
        self.used = 0
        self.rings_used = 0
        self.coronae_used = 0
        self.black_holes_used = 0

    def end_frame(self):
        """
        Hide what went unused, and sort what was drawn back to front.
        """

        drawn = self.pool[: self.used]

        order = sorted(range(len(drawn)), key=lambda i: -self._depth[i])

        for slot, index in enumerate(order):
            if self._bin_order[index] != slot:
                drawn[index].setBin("fixed", 20 + slot)
                self._bin_order[index] = slot

        for entity in self.pool[self.used :]:
            if entity.enabled:
                entity.enabled = False

        for glare in self.coronae[self.coronae_used :]:
            if glare.enabled:
                glare.enabled = False

        for ring in self.rings[self.rings_used :]:
            if ring.enabled:
                ring.enabled = False

        for hole in self.black_holes[self.black_holes_used :]:
            hole.hide()

    def _next(self):
        if self.used >= POOL_SIZE:
            return None

        entity = self.pool[self.used]
        self.used += 1

        entity.enabled = True

        return entity

    # --------------------------------------------------------
    # Solid bodies
    # --------------------------------------------------------

    def _place(self, entity, offset_metres, radius_metres, min_angular=None):
        """
        Position and size one body. Returns its distance, or None if
        it is too far to bother with.

        The distance compression is radial, so it preserves the
        *direction* of every point exactly and changes only its
        range. Angular size therefore has to be restored explicitly:
        a body of radius R at distance d subtends asin(R/d), and to
        subtend the same angle from the compressed distance p it has
        to be drawn with radius p sin(theta) = p R / d.

        Compressing the near and far edges separately instead - the
        obvious thing to try - shrinks bodies badly, by a factor of
        ten or more once they are many reference lengths away, since
        a logarithm's slope is nothing like its value.
        """

        distance = float(np.linalg.norm(offset_metres))

        if distance > DETAIL_RANGE_METRES:
            return None

        entity.position = compress_offset(offset_metres)

        self._depth[self.used - 1] = distance

        render_distance = compress_distance(distance)

        angle = min(radius_metres / distance, 0.985)

        floor = MIN_ANGULAR_RADIUS if min_angular is None else min_angular

        visual_radius = max(angle, floor) * render_distance

        entity.scale = float(visual_radius) * 2.0

        return distance

    def add_star(
        self,
        offset_metres,
        radius_metres,
        temperature_kelvin,
        luminosity_solar=1.0,
    ):
        """
        A star: a limb-darkened photosphere plus the glare its light
        makes in any real optics.

        How blinding the disc looks is set by temperature alone.
        Surface brightness is sigma T^4 and does not fall off with
        distance - only the angular size does - so the same
        expression covers a red dwarf and an O star, which is exactly
        the behaviour wanted.
        """

        entity = self._next()

        if entity is None:
            return None

        distance = self._place(entity, offset_metres, radius_metres)

        if distance is None:
            self.used -= 1
            entity.enabled = False
            return None

        temperature = max(float(temperature_kelvin), 300.0)

        rgb = temperature_to_rgb(temperature)

        entity.shader = star_shader()
        entity.set_shader_input(
            "star_colour", Vec4(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        )

        # Surface brightness relative to the Sun. The floor stands in
        # for dark adaptation: a red dwarf really is a hundred times
        # fainter per unit area than the Sun, but an eye looking at
        # nothing else would open up rather than see black.
        surface_brightness = (temperature / SUN_TEMPERATURE) ** 4

        entity.set_shader_input(
            "star_params",
            Vec2(float(np.clip(surface_brightness, 0.30, 4000.0)), LIMB_DARKENING),
        )

        self._add_glare(
            offset_metres, radius_metres, rgb, luminosity_solar, distance
        )

        return entity

    def _add_glare(self, offset_metres, radius_metres, rgb, luminosity_solar, distance):
        """
        The halo around a bright source.

        Its angular extent grows with the logarithm of the apparent
        flux, the same law the distant star field uses for its point
        sprites, so a star looks continuous as it crosses over from a
        sprite to a resolved disc.
        """

        if self.coronae_used >= len(self.coronae):
            return

        flux = (
            max(float(luminosity_solar), 1.0e-12)
            * SUN_LUMINOSITY
            / (4.0 * np.pi * distance * distance)
        )

        decades = np.log10(max(flux / GLARE_THRESHOLD_FLUX, 1.0))

        if decades <= 0.0:
            return

        star_angle = radius_metres / distance

        glare_angle = star_angle * 1.5 + GLARE_PER_DECADE * decades

        glare = self.coronae[self.coronae_used]
        self.coronae_used += 1

        render_distance = compress_distance(distance)

        glare.enabled = True
        glare.position = compress_offset(offset_metres)
        glare.scale = float(glare_angle * render_distance) * 2.0

        # Face the observer, who is always at the render-space origin.
        glare.look_at(Vec3(0.0, 0.0, 0.0))

        glare.set_shader_input(
            "corona_colour", Vec4(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        )
        glare.set_shader_input(
            "corona_params",
            Vec2(float(np.clip(0.30 * decades, 0.1, 5.0)), 0.42),
        )

    def add_planet(
        self,
        offset_metres,
        radius_metres,
        planet_type,
        star_offset_metres=None,
        star_rgb=(1.0, 1.0, 1.0),
        star_flux=1.0,
    ):
        """
        A planet, lit by its own star.

        The terminator is where the simulation's geometry becomes
        visible: the lit hemisphere always faces the star, so a
        planet on the far side of its orbit shows as a crescent.
        """

        entity = self._next()

        if entity is None:
            return None

        distance = self._place(entity, offset_metres, radius_metres)

        if distance is None:
            self.used -= 1
            entity.enabled = False
            return None

        rgb = PLANET_COLOURS.get(int(planet_type), (0.6, 0.6, 0.6))

        entity.shader = planet_shader()
        entity.set_shader_input(
            "planet_colour", Vec4(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        )

        if star_offset_metres is None:
            direction = -np.asarray(offset_metres, dtype=np.float64)
        else:
            direction = np.asarray(star_offset_metres, dtype=np.float64) - np.asarray(
                offset_metres, dtype=np.float64
            )

        norm = float(np.linalg.norm(direction))

        if norm > 0.0:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0, 1.0])

        # The shader works in view space, and the camera only ever
        # rotates, so the render-space direction transformed by the
        # camera's orientation is what it needs.
        view_dir = self._to_view_space(direction)

        entity.set_shader_input(
            "light_params",
            Vec4(float(view_dir[0]), float(view_dir[1]), float(view_dir[2]), 0.015),
        )

        entity.set_shader_input(
            "star_light",
            Vec4(
                float(star_rgb[0]),
                float(star_rgb[1]),
                float(star_rgb[2]),
                float(np.clip(star_flux, 0.05, 6.0)),
            ),
        )

        return entity

    @staticmethod
    def _to_view_space(direction):
        from ursina import camera

        quat = camera.getQuat(camera.getParent())
        quat.invertInPlace()

        vector = quat.xform(
            p3d.LVector3f(float(direction[0]), float(direction[1]), float(direction[2]))
        )

        return np.array([vector[0], vector[1], vector[2]])

    # --------------------------------------------------------
    # Black holes
    # --------------------------------------------------------

    def add_black_hole(
        self,
        offset_metres,
        mass_solar,
        spin=0.7,
        eddington_ratio=0.0,
        spin_axis=(0.0, 1.0, 0.0),
        cosmic_time=0.0,
    ):
        if self.black_holes_used >= len(self.black_holes):
            return None

        hole = self.black_holes[self.black_holes_used]
        self.black_holes_used += 1

        hole.show(
            offset_metres,
            mass_solar,
            spin,
            eddington_ratio,
            spin_axis,
            cosmic_time,
        )

        return hole

    def lens_sources(self):
        """
        The black holes currently drawn, for the lensing pass.
        """

        return [
            hole.lens
            for hole in self.black_holes[: self.black_holes_used]
            if hole.lens is not None
        ]

    # --------------------------------------------------------
    # Orbits
    # --------------------------------------------------------

    def add_orbit_ring(
        self, centre_offset_metres, radius_metres, inclination=0.0, node=0.0
    ):
        """
        A circle marking a planet's orbit.

        The compression is radial about the observer, so a circle in
        space is not a circle on screen: the near side is compressed
        less than the far side. Every vertex is therefore compressed
        individually.
        """

        if self.rings_used >= len(self.rings):
            return None

        ring = self.rings[self.rings_used]
        self.rings_used += 1

        cos_i, sin_i = np.cos(inclination), np.sin(inclination)
        cos_n, sin_n = np.cos(node), np.sin(node)

        x = _UNIT_CIRCLE[:, 0]
        y = _UNIT_CIRCLE[:, 1] * cos_i

        points = np.empty((RING_SEGMENTS, 3))

        points[:, 0] = (x * cos_n - y * sin_n) * radius_metres
        points[:, 2] = (x * sin_n + y * cos_n) * radius_metres
        points[:, 1] = _UNIT_CIRCLE[:, 1] * sin_i * radius_metres

        offsets = points + np.asarray(centre_offset_metres, dtype=np.float64)

        distances = np.linalg.norm(offsets, axis=1)

        compressed = offsets * (
            compress_distance(distances) / np.maximum(distances, 1.0)
        )[:, None]

        self.ring_buffers[self.rings_used - 1][:] = compressed

        ring.enabled = True
        ring.position = Vec3(0.0, 0.0, 0.0)
        ring.scale = 1.0
        ring.rotation = Vec3(0.0, 0.0, 0.0)

        return ring
