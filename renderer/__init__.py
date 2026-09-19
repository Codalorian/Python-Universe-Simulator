"""
The rendering layer.

Two facts drive every decision in here.

First, the universe spans sixty orders of magnitude and a linear scale
cannot show it. Every deep-space position is therefore compressed
logarithmically about the camera:

    r_render = log10(1 + r_lightyears) * VISUAL_SCALE

so a planet a light-minute away and a galaxy ten billion light years
away are both on screen at once, in the right order, with no depth
buffer precision problems.

Second, a quarter of a million objects cannot each be a scene graph
node. One Entity per star means one draw call per star and a frame
time measured in seconds. Instead every population is a single
interleaved GPU vertex buffer drawn as one point primitive, and the
compression above happens in the vertex shader, so flying through the
universe uploads nothing at all: the camera position is one uniform.
"""

import math

import numpy as np

import panda3d.core as p3d

from ursina import Entity, Mesh, Shader, Vec2, Vec3, Vec4, scene

from simulation_constants import C, YEAR, PARSEC, KPC, MPC, AU

LIGHT_YEAR = C * YEAR

# The reference length of the distance compression: one thousand
# kilometres.
#
# Everything closer than the reference is squeezed into the first
# render unit or so, so the reference sets the smallest thing that
# can be looked at. An astronomical unit leaves no room at all below
# planetary scale - a stellar-mass black hole's accretion disc is a
# millionth of an AU across and lands inside the near clip plane.
# A thousand kilometres puts the whole range, from a fifty-metre
# approach to the edge of the observable universe, inside 143 render
# units.
COMPRESSION_REFERENCE_M = 1.0e6

LIGHT_YEARS_IN_REFERENCE = LIGHT_YEAR / COMPRESSION_REFERENCE_M

# Render units per decade of distance.
VISUAL_SCALE = 7.0

# Floats per vertex: position (3), colour (4), log-luminosity and
# point-size gain (2).
VERTEX_STRIDE = 9

VERTEX_FORMAT = "p3f,c4f,t2f"


# ------------------------------------------------------------
# Shaders
# ------------------------------------------------------------

POINT_VERTEX_SHADER = """
#version 150

uniform mat4 p3d_ModelViewMatrix;

// Camera position, in the same units as the vertex positions.
uniform vec3 cam_pos;

// x: one position unit expressed in light years
// y: visual scale (render units per decade of distance)
// z: expansion scale factor a(t), applied to comoving positions
// w: global brightness gain
uniform vec4 space_params;

// x: pixels of point size per decade of brightness
// y: maximum point size in pixels
// z: faint cutoff, in log10 flux
// w: minimum point size in pixels
uniform vec4 size_params;

in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

out vec4 v_colour;
out float v_size;
out float v_core;

const float INV_LN10 = 0.434294481903;

// One light year, in units of the compression reference length.
const float LIGHT_YEARS_IN_REFERENCE = 9.4607305e9;

void main() {
    // Comoving position -> proper position -> light years from the
    // observer. Expansion is one multiply; travel is one subtract.
    vec3 offset = (p3d_Vertex.xyz * space_params.z - cam_pos) * space_params.x;

    float r = length(offset);

    // Logarithmic radial compression about the camera. This is what
    // puts an accretion disc and a galaxy ten billion light years
    // away on the same screen, in the right order, at once.
    float r_render =
        log(1.0 + r * LIGHT_YEARS_IN_REFERENCE) * INV_LN10 * space_params.y;

    vec3 compressed = offset * (r_render / max(r, 1e-12));

    gl_Position = p3d_ModelViewMatrix * vec4(compressed, 1.0);

    // Apparent brightness follows an inverse square law, so in
    // logarithms it is a subtraction. This is the astronomical
    // magnitude system: brighter objects get a wider point spread,
    // exactly as they do on a real detector.
    float log_flux = p3d_MultiTexCoord0.x - 2.0 * log(max(r, 1e-6)) * INV_LN10;

    float above_cutoff = log_flux - size_params.z;

    v_size = clamp(
        size_params.x * above_cutoff * p3d_MultiTexCoord0.y,
        size_params.w,
        size_params.y
    );

    // Fade the faintest objects out rather than letting them pop in.
    float fade = clamp(above_cutoff * 0.8, 0.0, 1.0);

    v_colour = vec4(p3d_Color.rgb, p3d_Color.a * fade * space_params.w);

    // Very bright objects get a saturated white core.
    v_core = clamp(above_cutoff * 0.16 - 0.5, 0.0, 0.85);
}
"""

# Each point becomes a screen-aligned quad here rather than relying on
# gl_PointSize. Hardware point sprites need GL_PROGRAM_POINT_SIZE,
# which is not reliably reachable through the scene graph, and they
# are capped at an implementation-defined maximum that is far too
# small for a nearby star. A geometry shader has neither limit.
POINT_GEOMETRY_SHADER = """
#version 150

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 p3d_ProjectionMatrix;
uniform vec2 window_size;

in vec4 v_colour[];
in float v_size[];
in float v_core[];

out vec2 g_uv;
out vec4 g_colour;
out float g_core;

void main() {
    if (v_colour[0].a <= 0.002) {
        return;
    }

    vec4 clip = p3d_ProjectionMatrix * gl_in[0].gl_Position;

    if (clip.w <= 0.0) {
        return;
    }

    // A width of v_size pixels, converted to clip space.
    vec2 half_size = vec2(
        v_size[0] / window_size.x,
        v_size[0] / window_size.y
    ) * clip.w;

    g_colour = v_colour[0];
    g_core = v_core[0];

    g_uv = vec2(-1.0, -1.0);
    gl_Position = clip + vec4(-half_size.x, -half_size.y, 0.0, 0.0);
    EmitVertex();

    g_uv = vec2(1.0, -1.0);
    gl_Position = clip + vec4(half_size.x, -half_size.y, 0.0, 0.0);
    EmitVertex();

    g_uv = vec2(-1.0, 1.0);
    gl_Position = clip + vec4(-half_size.x, half_size.y, 0.0, 0.0);
    EmitVertex();

    g_uv = vec2(1.0, 1.0);
    gl_Position = clip + vec4(half_size.x, half_size.y, 0.0, 0.0);
    EmitVertex();

    EndPrimitive();
}
"""

POINT_FRAGMENT_SHADER = """
#version 150

in vec2 g_uv;
in vec4 g_colour;
in float g_core;

out vec4 frag_colour;

void main() {
    float r2 = dot(g_uv, g_uv);

    if (r2 > 1.0) {
        discard;
    }

    // A Gaussian core plus a wider, fainter halo: the shape a real
    // stellar image takes once seeing and diffraction are folded in.
    float core = exp(-9.0 * r2);
    float halo = exp(-2.0 * r2) * 0.35;

    float intensity = (core + halo) * g_colour.a;

    // Saturated centres wash out towards white, as an overexposed
    // star does on any detector.
    vec3 rgb = mix(g_colour.rgb, vec3(1.0), g_core * core);

    frag_colour = vec4(rgb * intensity, intensity);
}
"""


def make_point_shader():
    return Shader(
        language=Shader.GLSL,
        vertex=POINT_VERTEX_SHADER,
        geometry=POINT_GEOMETRY_SHADER,
        fragment=POINT_FRAGMENT_SHADER,
    )


# ------------------------------------------------------------
# Scale helpers
# ------------------------------------------------------------

def compress_distance(metres):
    """
    The same mapping the vertex shader applies, for the handful of
    objects positioned on the CPU.

    Takes the plain-float path for scalars: this runs a few dozen
    times per frame on single values, where NumPy's per-call overhead
    costs far more than the arithmetic.
    """

    if isinstance(metres, (int, float)):
        return math.log10(1.0 + abs(metres) / COMPRESSION_REFERENCE_M) * VISUAL_SCALE

    scaled = np.abs(np.asarray(metres, dtype=np.float64)) / COMPRESSION_REFERENCE_M

    return np.log10(1.0 + scaled) * VISUAL_SCALE


def expand_distance(render_units):
    """
    Inverse of compress_distance, in metres.
    """

    r = np.abs(np.asarray(render_units, dtype=np.float64))

    return (10.0 ** (r / VISUAL_SCALE) - 1.0) * COMPRESSION_REFERENCE_M


def compress_offset(offset_metres):
    """
    Compress a 3-vector offset from the camera, preserving direction.
    """

    offset = np.asarray(offset_metres, dtype=np.float64)

    r = np.linalg.norm(offset)

    if r < 1.0e-12:
        return Vec3(0.0, 0.0, 0.0)

    scaled = offset * (compress_distance(r) / r)

    return Vec3(float(scaled[0]), float(scaled[1]), float(scaled[2]))


def format_distance(metres):
    """
    A distance in whatever unit a human would actually use.
    """

    d = float(abs(metres))

    if d < 1.0e3:
        return f"{d:,.0f} m"
    if d < 0.01 * AU:
        return f"{d / 1.0e3:,.0f} km"
    if d < 0.1 * LIGHT_YEAR:
        return f"{d / AU:,.3f} AU"
    if d < 1.0e3 * LIGHT_YEAR:
        return f"{d / LIGHT_YEAR:,.3f} light years"
    if d < 1.0e6 * PARSEC:
        return f"{d / KPC:,.2f} kpc"
    if d < 1.0e3 * MPC:
        return f"{d / MPC:,.2f} Mpc"

    return f"{d / MPC / 1000.0:,.3f} Gpc"


# ------------------------------------------------------------
# The GPU point batch
# ------------------------------------------------------------

class PointCloud:
    """
    One population of objects, drawn as a single point primitive.

    The vertex buffer is a NumPy array that this object owns; the same
    memory is handed to Panda3D. Writing to `self.data` and calling
    `upload()` (or `upload_rows()` for a sparse change) is the only
    cost of animating the whole population.
    """

    COLUMN_POSITION = slice(0, 3)
    COLUMN_COLOUR = slice(3, 7)
    COLUMN_LOG_LUMINOSITY = 7
    COLUMN_SIZE_GAIN = 8

    def __init__(
        self,
        count,
        unit_in_light_years,
        parent=None,
        shader=None,
        size_gain_pixels=3.0,
        max_point_size=64.0,
        min_point_size=1.0,
        faint_cutoff=-6.0,
        brightness=1.0,
    ):
        self.count = int(count)

        self.unit_in_light_years = float(unit_in_light_years)

        self.data = np.zeros((self.count, VERTEX_STRIDE), dtype=np.float32)

        # Default everything to invisible until the simulation fills it.
        self.data[:, self.COLUMN_SIZE_GAIN] = 1.0

        self.mesh = Mesh(
            vertex_buffer=self.data,
            vertex_buffer_length=self.count,
            vertex_buffer_format=VERTEX_FORMAT,
            triangles=np.arange(self.count, dtype=np.uint32),
            mode="point",
            static=False,
        )

        # Ursina treats parent=None as "detached", not "default", so a
        # cloud built without an explicit parent would never be drawn.
        self.entity = Entity(model=self.mesh, parent=parent or scene)

        # Take a permanent handle on the vertex array and keep a NumPy
        # view of it for the life of the cloud.
        #
        # Calling geom.modify_vertex_data() inside the render loop is
        # the obvious way to do this and it is a trap: the graphics
        # state guardian holds a reference to the vertex data it drew
        # last frame, so Panda3D copy-on-writes the whole buffer every
        # single call. At a quarter of a million vertices that leaked
        # sixteen megabytes per frame - a gigabyte every four seconds,
        # which will exhaust a machine's memory in well under a minute.
        #
        # Holding one handle means no copy is ever made, and writes
        # through it still reach the GPU.
        self._vertex_array = self.mesh.geom.modifyVertexData().modifyArray(0)

        self._gpu = np.frombuffer(
            memoryview(self._vertex_array).cast("B").cast("f"), dtype=np.float32
        ).reshape(self.count, VERTEX_STRIDE)

        self.entity.shader = shader if shader is not None else make_point_shader()

        # Additive blending: overlapping stars sum their light, which
        # is both physically right and what makes a dense star field
        # glow rather than flicker.
        self.entity.setAttrib(
            p3d.ColorBlendAttrib.make(
                p3d.ColorBlendAttrib.M_add,
                p3d.ColorBlendAttrib.O_one,
                p3d.ColorBlendAttrib.O_one,
            )
        )

        # Ursina puts every model into dual transparency mode, which
        # splits the draw into an alpha-tested opaque pass and a
        # sorted transparent one. That fights the explicit additive
        # blend above and drops the faint half of the population, so
        # it is turned off here and blending is left to the attribute
        # set just above.
        self.entity.setTransparency(p3d.TransparencyAttrib.M_none)
        self.mesh.setTransparency(p3d.TransparencyAttrib.M_none)

        # Additive light does not care what order it arrives in, and
        # the compressed depth range is far too wide for a depth
        # buffer to resolve, so both are simply switched off.
        self.entity.setDepthWrite(False)
        self.entity.setDepthTest(False)
        self.entity.setBin("fixed", 10)
        self.entity.setLightOff()

        self.brightness = float(brightness)

        self.size_params = Vec4(
            float(size_gain_pixels),
            float(max_point_size),
            float(faint_cutoff),
            float(min_point_size),
        )

        self.entity.set_shader_input("size_params", self.size_params)

        self.set_window_size()

        self.set_camera(np.zeros(3), scale_factor=1.0)

        self._dirty = True

    # --------------------------------------------------------
    # Buffer access
    # --------------------------------------------------------

    def upload(self):
        """
        Push the whole buffer. One memcpy of a few megabytes; cheap
        enough to do on a time jump, too expensive to do every frame.
        """

        self._gpu[:] = self.data

        self._dirty = False

    def upload_column(self, column):
        """
        Push a single interleaved column.

        Used where only one property animates - a population's
        visibility, say - so the other eight floats per vertex stay
        where they are.
        """

        self._gpu[:, column] = self.data[:, column]

    def upload_rows(self, indices):
        """
        Push only the rows that changed.

        Stellar evolution touches a handful of stars per frame out of a
        quarter of a million, so this is the path taken almost always.
        """

        if indices is None or len(indices) == 0:
            return

        self._gpu[indices] = self.data[indices]

    # --------------------------------------------------------
    # Per-frame uniforms
    # --------------------------------------------------------

    def set_camera(self, position, scale_factor=1.0):
        """
        Move the observer. This is the whole cost of travelling across
        the universe: three floats.
        """

        self.entity.set_shader_input(
            "cam_pos", Vec3(float(position[0]), float(position[1]), float(position[2]))
        )

        self.entity.set_shader_input(
            "space_params",
            Vec4(
                self.unit_in_light_years,
                VISUAL_SCALE,
                float(scale_factor),
                self.brightness,
            ),
        )

    def set_window_size(self, size=None):
        """
        The geometry shader sizes its quads in pixels, so it needs to
        know how big the window is. Call this when the window resizes.
        """

        if size is None:
            from ursina import window

            size = window.size

        self.entity.set_shader_input(
            "window_size", Vec2(max(float(size[0]), 1.0), max(float(size[1]), 1.0))
        )

    def set_brightness(self, value):
        self.brightness = float(value)

    def set_size_params(self, gain=None, maximum=None, cutoff=None, minimum=None):
        if gain is not None:
            self.size_params.x = float(gain)
        if maximum is not None:
            self.size_params.y = float(maximum)
        if cutoff is not None:
            self.size_params.z = float(cutoff)
        if minimum is not None:
            self.size_params.w = float(minimum)

        self.entity.set_shader_input("size_params", self.size_params)

    # --------------------------------------------------------
    # Convenience writers
    # --------------------------------------------------------

    def set_positions(self, positions):
        self.data[:, self.COLUMN_POSITION] = positions

    def set_colours(self, rgb, alpha=None):
        self.data[:, 3:6] = rgb

        if alpha is not None:
            self.data[:, 6] = alpha

    def set_alpha(self, alpha):
        self.data[:, 6] = alpha

    def set_log_luminosity(self, log_luminosity):
        self.data[:, self.COLUMN_LOG_LUMINOSITY] = log_luminosity

    def set_size_gain(self, gain):
        self.data[:, self.COLUMN_SIZE_GAIN] = gain

    @property
    def visible(self):
        return self.entity.enabled

    @visible.setter
    def visible(self, value):
        self.entity.enabled = bool(value)

    def destroy(self):
        from ursina import destroy

        destroy(self.entity)
