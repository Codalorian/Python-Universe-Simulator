"""
Gravitational lensing, as a screen-space pass.

A black hole bends the light of everything behind it. For a point
mass the deflection is

    alpha = 4 G M / (c^2 b)

for impact parameter b, and the lens equation that follows,

    beta = theta - theta_E^2 / theta,

says where on the sky a source at true position beta appears. Read
the other way round - which is what a renderer needs - it says that
the pixel at angular offset theta from the hole shows whatever lies
at beta. That inversion is exact for a Schwarzschild lens, needs no
iteration, and is one line of shader.

Inside sqrt(27) GM/c^2 no ray escapes at all, and that is the shadow:
not a drawn object but a region of the sky with nothing in it.

The scene therefore has to be rendered to a texture first, so this
pass has something to bend.
"""

import numpy as np

import panda3d.core as p3d

from direct.filter.FilterManager import FilterManager

# How many holes can bend light in one frame.
MAX_LENSES = 4

# Draw mask for geometry that must be drawn *after* the lensing pass
# rather than through it.
#
# An accretion disc is bent by the lens equation in its own vertex
# shader, because the screen-space pass cannot do it: the far side of
# the disc is occluded in the rasterised image before that pass ever
# sees it. Running both would deflect the disc twice, so it is put on
# a separate camera that renders over the finished, lensed frame.
OVERLAY_MASK = p3d.BitMask32.bit(7)


def make_overlay_camera(base, sort=60):
    """
    A second camera sharing the main view, drawn after the lensing
    filter, which sees only OVERLAY_MASK geometry.
    """

    camera = base.makeCamera(
        base.win, sort=sort, lens=base.camLens, camName="overlay_camera"
    )

    camera.reparentTo(base.cam)
    camera.setPosHpr(0, 0, 0, 0, 0, 0)

    camera.node().setCameraMask(OVERLAY_MASK)

    # Its display region must not wipe what the filter just drew.
    region = camera.node().getDisplayRegion(0)

    if region is not None:
        region.setClearColorActive(False)
        region.setClearDepthActive(True)

    # And the main camera must stop seeing the overlay geometry.
    #
    # A camera mask may not contain PandaNode's "overall" bit, which
    # is reserved for hiding a node from everything at once, so it is
    # cleared here as well as the overlay bit.
    base.cam.node().setCameraMask(
        p3d.BitMask32.allOn() & ~OVERLAY_MASK & ~p3d.PandaNode.getOverallBit()
    )

    return camera


def assign_to_overlay(node):
    """
    Make one node visible to the overlay camera alone.
    """

    node.hide(p3d.BitMask32.allOn())
    node.show(OVERLAY_MASK)

LENS_VERTEX = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 v_uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_uv = p3d_MultiTexCoord0;
}
"""

LENS_FRAGMENT = """
#version 150

uniform sampler2D tex;

// Per lens: xy = centre in normalised screen space,
//           z  = Einstein radius, w = shadow radius (same units).
uniform vec4 lenses[__MAX__];

uniform vec4 lens_count;   // x: how many entries are live
uniform vec4 screen;       // x: aspect ratio

in vec2 v_uv;

out vec4 frag_colour;

void main() {
    vec2 uv = v_uv;

    int count = int(lens_count.x);

    bool swallowed = false;

    // Light passing very close to the shadow is magnified so
    // violently that neighbouring pixels sample opposite sides of
    // the sky, which shows up as noise rather than as the multiple
    // images it really is. That annulus is faded instead, which is
    // also where a real image goes dark: rays there wind several
    // times around the hole and most of them fall in.
    float edge = 1.0;

    for (int i = 0; i < __MAX__; ++i) {
        if (i >= count) {
            break;
        }

        vec4 lens = lenses[i];

        // Work in a square angular space so the deflection stays
        // circular on a non-square window.
        vec2 offset = (uv - lens.xy) * vec2(screen.x, 1.0);

        float theta = length(offset);

        if (theta < 1e-6) {
            swallowed = true;
            break;
        }

        // Nothing escapes from inside the shadow.
        if (theta < lens.w) {
            swallowed = true;
            break;
        }

        edge *= smoothstep(lens.w, lens.w * 1.8, theta);

        // Lensing conserves surface brightness but not solid angle,
        // so an image is brightened by the magnification
        //
        //     mu = 1 / |1 - (theta_E / theta)^4|.
        //
        // Far from the hole that is 1 and nothing changes. At the
        // Einstein radius it diverges, which is why a perfectly
        // aligned source appears as a blazing ring. Inside it the
        // magnification is small: that is the faint secondary image,
        // and letting it stay at full brightness is what made the
        // middle of the frame look like noise rather than like a
        // second, dimmer view of the sky.
        float ratio = lens.z / theta;
        float quartic = ratio * ratio * ratio * ratio;

        edge *= clamp(1.0 / max(abs(1.0 - quartic), 0.08), 0.0, 6.0);

        // The lens equation, inverted: this pixel shows whatever sits
        // at beta. Light from directly behind the hole arrives from
        // every direction at once, which is why theta_E^2 / theta
        // diverges at the centre and sweeps the whole background into
        // a ring.
        float beta = theta - (lens.z * lens.z) / theta;

        vec2 direction = offset / theta;

        uv = lens.xy + direction * beta / vec2(screen.x, 1.0);
    }

    if (swallowed) {
        frag_colour = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Anything bent in from beyond the edge of the frame has no
    // record, so fall back to darkness rather than smearing the
    // border across the sky.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag_colour = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    frag_colour = vec4(texture(tex, uv).rgb * edge, 1.0);
}
""".replace("__MAX__", str(MAX_LENSES))


class LensingPass:
    """
    Renders the scene into a texture and bends it around whichever
    black holes are on screen.

    The pass costs a full-screen texture read, so it is switched off
    entirely whenever nothing is lensing.
    """

    def __init__(self, base, camera_node, lens_node):
        self.manager = FilterManager(base.win, camera_node)
        self.lens_node = lens_node

        self.texture = p3d.Texture()

        self.quad = self.manager.renderSceneInto(colortex=self.texture)

        # Straight to Panda rather than through Ursina's Shader
        # wrapper, which compiles lazily and expects to be attached
        # to an Entity.
        self.quad.setShader(
            p3d.Shader.make(p3d.Shader.SL_GLSL, LENS_VERTEX, LENS_FRAGMENT)
        )

        self.quad.setShaderInput("tex", self.texture)

        self._data = p3d.PTA_LVecBase4f()

        for _ in range(MAX_LENSES):
            self._data.pushBack(p3d.LVecBase4f(0.0, 0.0, 0.0, 0.0))

        self.quad.setShaderInput("lenses", self._data)
        self.quad.setShaderInput("lens_count", p3d.LVecBase4f(0, 0, 0, 0))
        self.quad.setShaderInput("screen", p3d.LVecBase4f(1.0, 0, 0, 0))

        self.active = False

    # --------------------------------------------------------

    def update(self, lenses, camera_node, lens, aspect_ratio, render_root):
        """
        lenses: the dicts produced by BlackHoleVisual, each carrying
        the offset to the hole and its angular scales.
        """

        live = 0

        for entry in lenses:
            if live >= MAX_LENSES:
                break

            screen_position = self._project(
                entry["offset"], camera_node, lens, render_root
            )

            if screen_position is None:
                continue

            # Angles on the sky converted to the same normalised
            # screen units the shader works in. The vertical field of
            # view spans one unit of v.
            vertical_fov = np.radians(lens.getVfov())

            scale = 1.0 / vertical_fov

            einstein = entry["einstein_angle"] * scale
            shadow = entry["shadow_angle"] * scale

            # Below a pixel or so there is nothing to see and the
            # distortion is not worth a full-screen pass.
            if einstein < 0.002 and shadow < 0.002:
                continue

            self._data.setElement(
                live,
                p3d.LVecBase4f(
                    float(screen_position[0]),
                    float(screen_position[1]),
                    float(einstein),
                    float(shadow),
                ),
            )

            live += 1

        self.quad.setShaderInput("lens_count", p3d.LVecBase4f(live, 0, 0, 0))
        self.quad.setShaderInput(
            "screen", p3d.LVecBase4f(float(aspect_ratio), 0, 0, 0)
        )

        self.active = live > 0

    @staticmethod
    def _project(offset_metres, camera_node, lens, render_root):
        """
        Where a direction lands on screen, as (u, v) in 0..1.
        """

        from renderer import compress_offset

        point = compress_offset(offset_metres)

        relative = camera_node.getRelativePoint(
            render_root, p3d.LPoint3f(point.x, point.y, point.z)
        )

        projected = p3d.LPoint2f()

        if not lens.project(relative, projected):
            return None

        return (projected[0] * 0.5 + 0.5, projected[1] * 0.5 + 0.5)
