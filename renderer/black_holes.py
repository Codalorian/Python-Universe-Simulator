"""
Drawing black holes.

Four things make a black hole legible, and all four are computed from
its mass, spin and accretion rate rather than chosen by eye:

  * the shadow, sqrt(27) GM/c^2 across, which is what a distant
    observer actually sees rather than the Schwarzschild radius;
  * the photon ring just outside it, where light orbits the hole;
  * an accretion disc, if the hole is feeding, coloured by the
    Shakura-Sunyaev temperature profile and beamed towards the
    observer on the side rotating their way;
  * a pair of jets along the spin axis, powered by the Blandford-
    Znajek process, reaching far beyond the galaxy that made them.

Gravitational lensing of everything behind is handled separately, as
a screen-space pass, since it has to distort pixels this renderer has
already drawn.
"""

import numpy as np

import panda3d.core as p3d

from ursina import Entity, Vec2, Vec3, Vec4, scene
from ursina.shaders import unlit_shader

from simulation_constants import C, G, SUN_MASS, PARSEC, KPC, YEAR

from compact_objects.black_holes import (
    gravitational_radius,
    shadow_radius,
    photon_sphere_radius,
    eddington_luminosity_watts,
    radiative_efficiency,
)

from compact_objects.quasars import disc_temperature, jet_power

from stellar_pop.stellar_evolution.relations import isco_radius
from stellar_pop.stellar_evolution.blackbody import temperature_to_rgb

from renderer import (
    compress_distance,
    compress_offset,
    VISUAL_SCALE,
    COMPRESSION_REFERENCE_M,
)
from renderer.geometry import (
    sphere_model,
    annulus_model,
    jet_model,
    billboard_model,
)
from renderer.shaders import disc_shader, jet_shader, shadow_shader

# Outer edge of the drawn disc, in gravitational radii. Real discs
# extend much further but fade below anything visible.
DISC_OUTER_RG = 34.0

# Resolution of the radial colour ramp handed to the disc shader.
RAMP_SIZE = 256


def disc_colour_ramp(mass_solar, accretion_rate_kg_s, spin=0.0):
    """
    A one-dimensional texture of the disc's colour against radius.

    T(r) comes straight from the Shakura-Sunyaev thin-disc solution,
    and the colour from the CIE integral of that blackbody. A
    stellar-mass hole runs at 1e7 K and glows blue-white into the
    X-ray; a billion-solar-mass quasar disc peaks near 1e5 K, because
    the temperature falls as M^-1/4.
    """

    r_in = float(isco_radius(mass_solar, spin))
    r_out = DISC_OUTER_RG * float(gravitational_radius(mass_solar))

    fraction = np.linspace(1.0 / RAMP_SIZE, 1.0, RAMP_SIZE)

    radii = np.maximum(fraction * r_out, r_in * 1.0001)

    temperature = disc_temperature(radii, mass_solar, accretion_rate_kg_s, spin)

    rgb = temperature_to_rgb(np.clip(temperature, 800.0, 120_000.0))

    # Colour only: the radial brightness is evaluated in the shader,
    # because its four-decade range does not fit in eight bits.
    values = np.clip(rgb, 0.0, 1.0)

    texture = p3d.Texture("disc_ramp")
    texture.setup2dTexture(
        RAMP_SIZE, 1, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgb8
    )

    data = (values * 255.0).astype(np.uint8)

    # Panda stores texture rows bottom-up and in BGR order.
    texture.setRamImage(data[:, ::-1].tobytes())

    texture.setWrapU(p3d.Texture.WM_clamp)
    texture.setWrapV(p3d.Texture.WM_clamp)
    texture.setMagfilter(p3d.Texture.FT_linear)

    return texture


def jet_length_metres(power_watts):
    """
    How far a jet reaches.

    A jet drills a cavity through the surrounding medium, so its
    length grows with the cube root of its power. Calibrated to the
    few-kiloparsec scale of a quasar jet at 1e38 W, which puts a
    stellar-mass microquasar's jets at a few parsecs.
    """

    power = max(float(power_watts), 1.0)

    return 3.0 * KPC * (power / 1.0e38) ** (1.0 / 3.0)


def _orthonormal_frame(axis):
    """
    A unit vector across the given axis, for the shaders that build
    an oriented body vertex by vertex.
    """

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)

    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[1]) > 0.9 else np.array([0.0, 1.0, 0.0])

    right = np.cross(seed, axis)
    right = right / max(float(np.linalg.norm(right)), 1.0e-12)

    return axis, right


def _orient_to_axis(entity, axis):
    """
    Point an entity's +Y at a direction given in render space.
    """

    axis = np.asarray(axis, dtype=np.float64)

    norm = np.linalg.norm(axis)

    if norm < 1.0e-12:
        return

    axis = axis / norm

    up = np.array([0.0, 1.0, 0.0])

    cross = np.cross(up, axis)
    dot = float(np.clip(np.dot(up, axis), -1.0, 1.0))

    if np.linalg.norm(cross) < 1.0e-9:
        if dot > 0.0:
            entity.setQuat(p3d.LQuaternionf(1, 0, 0, 0))
        else:
            entity.setQuat(p3d.LQuaternionf(0, 1, 0, 0))
        return

    cross = cross / np.linalg.norm(cross)

    quat = p3d.LQuaternionf()
    quat.setFromAxisAngleRad(
        float(np.arccos(dot)), p3d.LVector3f(*(float(v) for v in cross))
    )

    entity.setQuat(quat)


class BlackHoleVisual:
    """
    One black hole's set of scene nodes, reused frame to frame.
    """

    def __init__(self, parent=None, overlay=False):
        parent = parent or scene

        # The shadow is a screen-facing disc, not a sphere.
        #
        # It has to be exactly circular and exactly sqrt(27) GM/c^2
        # across, and it has to sit on top of the disc: the lensed
        # disc mesh is stretched so violently near the hole that any
        # boundary derived from its own geometry comes out as a
        # ragged polygon. A billboard cut to a circle in the fragment
        # shader has no such problem, and it is the truthful shape -
        # a region of the sky from which nothing returns.
        self.horizon = Entity(model=billboard_model(), parent=parent, enabled=False)
        self.horizon.shader = shadow_shader()
        self.horizon.setTwoSided(True)
        self.horizon.color = Vec4(0.0, 0.0, 0.0, 1.0)
        self.horizon.setLightOff()
        self.horizon.setTransparency(p3d.TransparencyAttrib.M_none)
        self.horizon.model.setTransparency(p3d.TransparencyAttrib.M_none)
        # The shadow and the disc are the one pair in the scene that
        # genuinely need sorting against each other: the near half of
        # the disc passes in front of the hole and the far half goes
        # behind it. They sit at almost the same range, so the depth
        # buffer resolves them even with the frustum this wide.
        self.horizon.setDepthWrite(False)
        self.horizon.setDepthTest(False)
        self.horizon.setBin("fixed", 44)

        # Two images of the same disc.
        #
        # The primary is the one that matters: the lens equation
        # pushes every patch of disc outward from the hole, and the
        # far side - which ought to be hidden behind it - is lifted up
        # over the top, so the entire upper surface is visible from
        # any viewing angle. The secondary is the faint, heavily
        # squashed view of the disc's *underside* that hugs the
        # shadow, made of rays that loop around the hole and climb
        # back out.
        #
        # Both are bent in the vertex shader rather than by the
        # screen-space pass, which cannot do this: the far side is
        # occluded in the rasterised image before that pass ever sees
        # it. They are therefore drawn by the overlay camera, which
        # renders after the lensing filter, so nothing is deflected
        # twice.
        self.disc = self._make_disc(parent, side=1.0, order=42)
        self.disc_secondary = self._make_disc(parent, side=-1.0, order=43)

        self.discs = (self.disc, self.disc_secondary)

        if overlay:
            from renderer.lensing import assign_to_overlay

            # The horizon goes with them. The lensing pass already
            # blacks out everything inside the shadow angle, so
            # sending the sphere through the filter as well left two
            # disagreeing versions of the same shadow - one crisp and
            # one smeared outward - which read as a ragged blob.
            for node in (self.horizon, *self.discs):
                assign_to_overlay(node)

        self.jets = []

        for _ in range(2):
            jet = Entity(model=jet_model(), parent=parent, enabled=False)
            jet.shader = jet_shader()
            jet.setLightOff()
            jet.setTransparency(p3d.TransparencyAttrib.M_none)
            jet.model.setTransparency(p3d.TransparencyAttrib.M_none)
            jet.setDepthWrite(False)
            jet.setDepthTest(False)
            jet.setTwoSided(True)
            jet.setBin("fixed", 40)
            jet.setAttrib(
                p3d.ColorBlendAttrib.make(
                    p3d.ColorBlendAttrib.M_add,
                    p3d.ColorBlendAttrib.O_one,
                    p3d.ColorBlendAttrib.O_one,
                )
            )
            self.jets.append(jet)

        self._ramp_key = None
        self._ramp = None

    def _make_disc(self, parent, side, order):
        # The inner edge sits at the innermost stable circular orbit,
        # 6 GM/c^2 for a non-spinning hole.
        disc = Entity(
            model=annulus_model(
                inner_fraction=6.0 / DISC_OUTER_RG, rings=96, segments=256
            ),
            parent=parent,
            enabled=False,
        )

        disc.shader = disc_shader()
        disc.setLightOff()

        # Ursina puts every model into dual transparency mode, which
        # alpha-tests and re-sorts; against an additive blend that
        # turns the disc into a dark cut-out instead of light.
        disc.setTransparency(p3d.TransparencyAttrib.M_none)
        disc.model.setTransparency(p3d.TransparencyAttrib.M_none)

        disc.setDepthWrite(False)
        disc.setDepthTest(False)
        disc.setTwoSided(True)
        disc.setBin("fixed", order)
        disc.setAttrib(
            p3d.ColorBlendAttrib.make(
                p3d.ColorBlendAttrib.M_add,
                p3d.ColorBlendAttrib.O_one,
                p3d.ColorBlendAttrib.O_one,
            )
        )

        disc.side = side

        return disc

        # Filled in by show(), read by the lensing pass.
        self.lens = None

    # --------------------------------------------------------

    def hide(self):
        for node in (self.horizon, *self.discs, *self.jets):
            if node.enabled:
                node.enabled = False

        self.lens = None

    def show(
        self,
        offset_metres,
        mass_solar,
        spin,
        eddington_ratio,
        spin_axis,
        cosmic_time,
    ):
        """
        Place and light one black hole.

        offset_metres is the vector from the observer, in metres and
        universe axes; spin_axis likewise. eddington_ratio of zero
        means the hole is quiescent: no disc, no jets, just a shadow
        and whatever it does to the light behind it.
        """

        offset = np.asarray(offset_metres, dtype=np.float64)

        distance = float(np.linalg.norm(offset))

        if distance < 1.0:
            self.hide()
            return

        r_g = float(gravitational_radius(mass_solar))
        r_shadow = float(shadow_radius(mass_solar))

        position = compress_offset(offset)

        render_distance = compress_distance(distance)

        # The compression preserves directions, not ranges, so a
        # rigidly scaled body has to have its angular size restored:
        # radius p sin(theta) at compressed distance p subtends the
        # same angle theta the body really does.
        def visual_radius(radius_m):
            angle = min(radius_m / distance, 0.985)

            return max(angle * render_distance, 1.0e-7)

        # --- the shadow ---
        horizon_radius = visual_radius(r_shadow)

        self.horizon.enabled = True
        self.horizon.position = position
        self.horizon.scale = horizon_radius * 2.0
        self.horizon.look_at(Vec3(0.0, 0.0, 0.0))

        # --- what the lensing pass needs ---
        self.lens = {
            "offset": offset,
            "distance": distance,
            "shadow_angle": r_shadow / distance,
            # Einstein radius for a source far behind the hole:
            # theta_E = sqrt(4GM / (c^2 d)) = sqrt(2 Rs / d).
            "einstein_angle": float(np.sqrt(4.0 * r_g / distance)),
        }

        if eddington_ratio <= 0.0:
            for jet in self.jets:
                jet.enabled = False
            for disc in self.discs:
                disc.enabled = False
            return

        # --- accretion disc ---
        spin = float(np.clip(spin, 0.0, 0.998))

        efficiency = float(radiative_efficiency(spin))

        luminosity = eddington_ratio * float(eddington_luminosity_watts(mass_solar))
        accretion_rate = luminosity / (efficiency * C * C)

        key = (round(float(mass_solar), 3), round(float(eddington_ratio), 4))

        if key != self._ramp_key:
            # setTexture rather than Ursina's texture property, which
            # expects its own wrapper type rather than a raw Panda
            # texture.
            self._ramp = disc_colour_ramp(mass_solar, accretion_rate, spin)

            for disc in self.discs:
                disc.setTexture(self._ramp, 1)

            self._ramp_key = key

        r_out = DISC_OUTER_RG * r_g

        r_in_fraction = float(isco_radius(mass_solar, spin)) / r_out

        axis, right = _orthonormal_frame(spin_axis)

        offset_scaled = offset / COMPRESSION_REFERENCE_M

        shadow_angle = r_shadow / distance

        for disc in self.discs:
            disc.enabled = True

            # The shader places every vertex itself, so the node's own
            # transform stays at identity.
            disc.position = Vec3(0.0, 0.0, 0.0)
            disc.scale = 1.0

            self._set_body_frame(
                disc, offset_scaled, axis, right, r_out / COMPRESSION_REFERENCE_M
            )

            disc.set_shader_input(
                "lens_geometry",
                Vec4(
                    # 4 GM / c^2, the coefficient in the Einstein
                    # radius, in the same units as everything else.
                    float(4.0 * r_g / COMPRESSION_REFERENCE_M),
                    float(disc.side),
                    float(shadow_angle),
                    0.0,
                ),
            )

        # Orbital speed at the outer edge, as a fraction of c. The
        # shader scales it as 1/sqrt(r) to get the rest of the disc.
        beta_outer = float(np.sqrt(G * mass_solar * SUN_MASS / r_out) / C)

        # How edge-on the disc is: beaming only shows when some of the
        # orbital motion points at the observer.
        inclination = float(
            np.sqrt(max(0.0, 1.0 - np.dot(axis, offset / distance) ** 2))
        )

        for disc in self.discs:
            disc.set_shader_input(
                "disc_profile",
                Vec2(float(np.clip(r_in_fraction, 0.01, 0.5)), 0.55),
            )

            disc.set_shader_input(
                "disc_params",
                Vec4(
                    # With a Reinhard response there is headroom to
                    # make the disc as bright as it should be without
                    # losing the radial gradient or the beaming.
                    4.5,
                    beta_outer,
                    inclination,
                    float(cosmic_time / YEAR % 1.0e6),
                ),
            )

        # --- jets ---
        power = float(jet_power(mass_solar, eddington_ratio, max(spin, 0.3)))

        length = jet_length_metres(power)

        # A jet is collimated within a few tens of gravitational radii
        # and then opens out at a degree or so, which is what the
        # observed ones do over kiloparsec scales.
        base_radius = 12.0 * r_g / COMPRESSION_REFERENCE_M
        opening_angle = 0.022

        for sign, jet in zip((1.0, -1.0), self.jets):
            jet.enabled = True
            jet.position = Vec3(0.0, 0.0, 0.0)
            jet.scale = 1.0

            self._set_body_frame(
                jet,
                offset_scaled,
                axis * sign,
                right,
                length / COMPRESSION_REFERENCE_M,
            )

            jet.set_shader_input(
                "jet_shape",
                Vec4(
                    float(base_radius),
                    float(opening_angle),
                    float(length / COMPRESSION_REFERENCE_M),
                    0.0,
                ),
            )

            jet.set_shader_input("jet_colour", Vec4(0.62, 0.76, 1.0, 1.0))
            jet.set_shader_input(
                "jet_params",
                # Well below the disc. A jet is optically thin
                # synchrotron emission; in visible light the disc
                # outshines it by orders of magnitude, and letting
                # the jet dominate hides the thing worth looking at.
                Vec4(0.42, float(cosmic_time / YEAR % 1.0e6), 1.0, 0.0),
            )

    @staticmethod
    def _set_body_frame(entity, offset, axis, right, size):
        entity.set_shader_input(
            "body_offset",
            Vec4(
                float(offset[0]),
                float(offset[1]),
                float(offset[2]),
                float(size),
            ),
        )
        entity.set_shader_input(
            "body_axis",
            Vec4(float(axis[0]), float(axis[1]), float(axis[2]), 0.0),
        )
        entity.set_shader_input(
            "body_right",
            Vec4(float(right[0]), float(right[1]), float(right[2]), 0.0),
        )
        entity.set_shader_input("space_scale", Vec2(VISUAL_SCALE, 0.0))
