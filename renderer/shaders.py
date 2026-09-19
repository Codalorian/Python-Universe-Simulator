"""
Surface shaders for resolved bodies.

Each one is driven by the same quantities the simulation already
computes, so nothing here is a hand-tuned look: a star's brightness
comes from its effective temperature through Stefan-Boltzmann, a
planet's shading from the direction of its own star, and an accretion
disc's colour from the Shakura-Sunyaev temperature profile.
"""

from ursina import Shader

# ------------------------------------------------------------
# Shared fragments
# ------------------------------------------------------------
#
# A detector saturates. Doubling the light on an already-bright pixel
# does not double what you see, it pushes the colour towards white,
# and that is what makes something read as luminous rather than
# merely pale. This is the standard exponential response.

_TONEMAP = """
vec3 tonemap(vec3 value) {
    return vec3(1.0) - exp(-value);
}
"""

# A sphere centred on its own origin has the normalised vertex
# position as its surface normal, so smooth shading never depends on
# how finely the model happens to be tessellated.
_SPHERE_VERTEX = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;

in vec4 p3d_Vertex;

out vec3 v_normal;
out vec3 v_eye;
out vec3 v_object;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;

    v_object = normalize(p3d_Vertex.xyz);
    v_normal = normalize(mat3(p3d_ModelViewMatrix) * v_object);
    v_eye = -(p3d_ModelViewMatrix * p3d_Vertex).xyz;
}
"""


# ------------------------------------------------------------
# Stars
# ------------------------------------------------------------

STAR_FRAGMENT = """
#version 150

// rgb: the star's CIE colour. a: unused.
uniform vec4 star_colour;

// x: surface brightness relative to the Sun, (T/Tsun)^4
// y: limb darkening coefficient
uniform vec2 star_params;

in vec3 v_normal;
in vec3 v_eye;

out vec4 frag_colour;

__TONEMAP__

void main() {
    float mu = clamp(dot(normalize(v_normal), normalize(v_eye)), 0.0, 1.0);

    // Linear limb darkening: light leaving the limb escapes from
    // higher, cooler layers than light leaving the centre.
    float darkening = 1.0 - star_params.y * (1.0 - mu);

    // Surface brightness is sigma T^4 and does not fall off with
    // distance - only the angular size does. So how blinding a
    // resolved star looks is set by its temperature alone, and the
    // same expression covers a red dwarf and an O star.
    vec3 warm = star_colour.rgb * vec3(1.08, 0.90, 0.74);
    vec3 rgb = mix(warm, star_colour.rgb, mu);

    frag_colour = vec4(tonemap(rgb * star_params.x * darkening), 1.0);
}
""".replace("__TONEMAP__", _TONEMAP)


# The glare around a bright source: not part of the star, but what
# any real optics do with its light. The radial profile is the same
# one the distant star field uses for its point sprites, so a star
# looks the same crossing over from a sprite to a resolved disc.
CORONA_VERTEX = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;

out vec2 v_uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_uv = p3d_Vertex.xy * 2.0;
}
"""

CORONA_FRAGMENT = """
#version 150

uniform vec4 corona_colour;

// x: overall gain, y: how far the halo reaches
uniform vec2 corona_params;

in vec2 v_uv;

out vec4 frag_colour;

void main() {
    float r = length(v_uv);

    if (r > 1.0) {
        discard;
    }

    // A tight core plus a broad halo, the shape of a real stellar
    // image once diffraction and scattering are included.
    //
    // Each term has its own value at the rim subtracted, so the
    // profile reaches exactly zero at the edge of the billboard.
    // Without that the halo is cut off partway down and the quad's
    // own boundary shows as a hard ring around the star.
    float r2 = r * r;

    float core = (exp(-16.0 * r2) - exp(-16.0)) / (1.0 - exp(-16.0));
    float halo = (exp(-3.4 * r2) - exp(-3.4)) / (1.0 - exp(-3.4));

    float intensity = (core + halo * corona_params.y) * corona_params.x;

    vec3 rgb = mix(corona_colour.rgb, vec3(1.0), clamp(core * 1.4, 0.0, 1.0));

    frag_colour = vec4(rgb * intensity, intensity);
}
"""


# ------------------------------------------------------------
# Planets
# ------------------------------------------------------------

PLANET_FRAGMENT = """
#version 150

uniform vec4 planet_colour;

// xyz: direction to the parent star, in view space
// w:   ambient fraction
uniform vec4 light_params;

// rgb: the star's colour, a: its apparent brightness here
uniform vec4 star_light;

in vec3 v_normal;
in vec3 v_eye;

out vec4 frag_colour;

__TONEMAP__

void main() {
    vec3 normal = normalize(v_normal);

    float lambert = dot(normal, normalize(light_params.xyz));

    // A soft terminator: a planet is lit by a star of finite angular
    // size, so the day-night boundary is a gradient, not a line.
    float day = smoothstep(-0.08, 0.14, lambert);

    vec3 lit = planet_colour.rgb * star_light.rgb * star_light.a * day;

    vec3 ambient = planet_colour.rgb * light_params.w;

    // Limb brightening towards the terminator, standing in for the
    // forward scattering a real atmosphere does.
    float mu = clamp(dot(normal, normalize(v_eye)), 0.0, 1.0);
    float rim = pow(1.0 - mu, 3.0) * day * 0.35;

    frag_colour = vec4(tonemap(lit + ambient + planet_colour.rgb * rim), 1.0);
}
""".replace("__TONEMAP__", _TONEMAP)


# ------------------------------------------------------------
# Accretion discs
# ------------------------------------------------------------

# Bodies that are large compared with how far away they are - an
# accretion disc seen from a few hundred gravitational radii, a jet
# that is ten billion times longer than the observer's distance from
# its base - cannot be drawn as rigid shapes. The distance
# compression is radial about the observer, so the near end of such a
# body is compressed far less than the far end, and a rigid mesh
# comes out the wrong shape entirely.
#
# These shaders therefore build each vertex's true offset from the
# observer in astronomical units and compress it individually, the
# same way the point clouds do. The entity's own transform is left at
# identity.
_ORIENTED_BODY = """
// xyz: observer -> body centre, in compression reference lengths.
// w:   model units -> the same.
uniform vec4 body_offset;

// Orthonormal frame: the spin axis, and one vector across it.
uniform vec4 body_axis;
uniform vec4 body_right;

uniform vec2 space_scale;   // x: render units per decade

const float INV_LN10 = 0.434294481903;

vec3 compress(vec3 offset) {
    float r = length(offset);

    float r_render = log(1.0 + r) * INV_LN10 * space_scale.x;

    return offset * (r_render / max(r, 1e-9));
}

vec3 body_world(vec3 local) {
    vec3 axis = normalize(body_axis.xyz);
    vec3 right = normalize(body_right.xyz);
    vec3 forward = cross(axis, right);

    vec3 in_place = (right * local.x + axis * local.y + forward * local.z);

    return body_offset.xyz + in_place * body_offset.w;
}
"""

DISC_VERTEX = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// x: 4 GM / c^2, in reference lengths
// y: +1 for the primary image, -1 for the secondary
// z: angular radius of the shadow, seen from the observer
uniform vec4 lens_geometry;

out vec2 v_uv;
out float v_magnification;
out vec3 v_image_dir;

__ORIENTED__

void main() {
    // Where this patch of disc actually is, before any bending.
    vec3 world = body_world(p3d_Vertex.xyz);

    vec3 to_hole = body_offset.xyz;

    float lens_distance = length(to_hole);
    float source_distance = length(world);

    vec3 hole_dir = to_hole / max(lens_distance, 1e-12);
    vec3 source_dir = world / max(source_distance, 1e-12);

    float cos_beta = clamp(dot(source_dir, hole_dir), -1.0, 1.0);

    // True angular separation of this patch from the hole.
    float beta = acos(cos_beta);

    // How far behind the hole the patch lies, along the line of
    // sight. Negative means it is in front, where nothing bends it.
    float behind = dot(world - to_hole, hole_dir);

    float einstein_sq = 0.0;

    if (behind > 0.0) {
        einstein_sq =
            lens_geometry.x * behind / max(lens_distance * source_distance, 1e-20);
    }

    float side = lens_geometry.y;

    float magnification = 0.0;
    float theta = beta;

    if (einstein_sq > 0.0) {
        // The lens equation beta = theta - theta_E^2 / theta, solved
        // for the image position. Two roots: the primary image
        // pushed outward from the hole, and a secondary one on the
        // opposite side, inside the Einstein radius. It is the
        // primary root that lifts the far side of the disc up over
        // the top of the hole, so the whole upper surface is visible
        // from any angle.
        float root = sqrt(beta * beta + 4.0 * einstein_sq);

        theta = 0.5 * (beta + side * root);

        // Lensing conserves surface brightness but not solid angle.
        float u = max(beta / sqrt(einstein_sq), 1e-4);

        magnification =
            0.5 * ((u * u + 2.0) / (u * sqrt(u * u + 4.0)) + side);
    } else if (side > 0.0) {
        // Unlensed: seen directly, at unit magnification.
        magnification = 1.0;
    }

    v_magnification = clamp(magnification, 0.0, 14.0);

    // Rotate the viewing direction from beta to theta, in the plane
    // containing the hole and the patch.
    vec3 across = source_dir - hole_dir * cos_beta;

    float across_length = length(across);

    vec3 across_dir = across_length > 1e-9
        ? across / across_length
        : normalize(cross(hole_dir, vec3(0.0, 1.0, 0.0)));

    vec3 image_dir = hole_dir * cos(theta) + across_dir * sin(theta);

    // The range is left alone, so the distance compression still
    // places it at the right depth.
    gl_Position = p3d_ModelViewProjectionMatrix
                * vec4(compress(image_dir * source_distance), 1.0);

    // Handed to the fragment stage so the shadow can be cut out per
    // pixel. Testing an interpolated *angle* instead gives a visibly
    // polygonal hole: the lensing stretches the mesh so violently
    // near the hole that linear interpolation between vertices is
    // nothing like the real boundary.
    v_image_dir = image_dir;

    v_uv = p3d_MultiTexCoord0;
}
""".replace("__ORIENTED__", _ORIENTED_BODY)


DISC_FRAGMENT = """
#version 150

// A one-dimensional ramp of the disc's blackbody colour against
// radius, built on the CPU from the Shakura-Sunyaev profile.
uniform sampler2D p3d_Texture0;

// x: overall brightness
// y: orbital speed at the outer edge, as a fraction of c
// z: how edge-on the disc is
// w: time, for the flow of material around the disc
uniform vec4 disc_params;

// x: inner edge as a fraction of the outer radius (the ISCO)
// y: display gamma for the radial brightness
uniform vec2 disc_profile;

// Repeated from the vertex stage: the fragment needs the hole's
// direction and angular size to cut out the shadow.
uniform vec4 body_offset;
uniform vec4 lens_geometry;

in vec2 v_uv;
in float v_magnification;
in vec3 v_image_dir;

out vec4 frag_colour;

// The exponential response saturates hard, which is right for a
// stellar photosphere but wrong here: an accretion disc's whole
// story is its brightness *contrast* - four decades of radial
// falloff and an order of magnitude of Doppler beaming between one
// side and the other. Under 1 - exp(-x) all of that clips to flat
// white. Reinhard's x/(1+x) never quite reaches one, so the
// ordering survives all the way up.
vec3 disc_tonemap(vec3 value) {
    return value / (vec3(1.0) + value);
}

void main() {
    if (v_magnification < 0.015) {
        discard;
    }

    // Light whose image lands inside the shadow never escaped.
    float from_hole = acos(
        clamp(dot(normalize(v_image_dir), normalize(body_offset.xyz)), -1.0, 1.0)
    );

    if (from_hole < lens_geometry.z) {
        discard;
    }

    float radius = clamp(v_uv.x, 0.001, 1.0);

    // Colour comes from the ramp, which holds the true blackbody
    // colour of the Shakura-Sunyaev temperature at each radius.
    vec3 rgb = texture(p3d_Texture0, vec2(radius, 0.5)).rgb;

    // Brightness is evaluated here rather than baked into the ramp.
    // Flux goes as r^-3 (1 - sqrt(r_in/r)), which spans four orders
    // of magnitude across the disc - far more than an eight-bit
    // texture holds, and far more than a screen can show at once.
    // The gamma below compresses that range the same way every
    // published image of an accretion disc does.
    float inner = disc_profile.x;

    float flux = pow(radius, -3.0) * (1.0 - sqrt(inner / max(radius, inner)));

    // Normalised to the peak, which the thin-disc solution puts at
    // (49/36) r_in.
    float peak_r = 1.3611 * inner;
    float peak = pow(peak_r, -3.0) * (1.0 - sqrt(inner / peak_r));

    float brightness = pow(clamp(flux / max(peak, 1e-12), 0.0, 1.0), disc_profile.y);

    // Keplerian speed falls as 1/sqrt(r), so the inner disc is
    // relativistic and the outer disc is not.
    float beta = disc_params.y / sqrt(radius);

    // Relativistic beaming: material coming towards the observer is
    // brighter by roughly the Doppler factor cubed. This is why one
    // side of an accretion disc outshines the other.
    float along = sin(v_uv.y * 6.2831853);
    float doppler = 1.0 / (1.0 - beta * along * disc_params.z);
    // Cubed, as relativistic beaming goes for a continuum source.
    float beaming = pow(clamp(doppler, 0.1, 6.0), 3.0);

    // Turbulence in the flow, carried around at the orbital rate.
    float phase = v_uv.y * 6.2831853 - disc_params.w / pow(radius, 1.5);
    float mottle = 0.90 + 0.10 * sin(phase * 4.0) * sin(phase * 2.0 + radius * 9.0);

    // The disc thins and fades at its outer edge.
    float fade = smoothstep(1.0, 0.86, radius);

    // And at its inner edge. Material does not stop dead at the
    // innermost stable orbit; inside it, it plunges. Fading there
    // also softens the boundary of the lensed image: the Einstein
    // radius varies steeply across the inner disc, and evaluating it
    // once per vertex leaves that boundary visibly faceted.
    fade *= smoothstep(inner, inner * 2.1, radius);

    float intensity =
        disc_params.x * brightness * beaming * mottle * fade * v_magnification;

    frag_colour = vec4(disc_tonemap(rgb * intensity), 1.0);
}
""".replace("__TONEMAP__", _TONEMAP)


# ------------------------------------------------------------
# Relativistic jets
# ------------------------------------------------------------

JET_VERTEX = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// x: base radius, y: opening half-angle in radians, z: length.
// Radii and length in reference lengths.
uniform vec4 jet_shape;

out vec2 v_uv;

__ORIENTED__

void main() {
    float along = p3d_Vertex.y;

    // Collimated within tens of gravitational radii, then a cone of
    // fixed opening angle - which is what a jet actually is once it
    // has left the launching region.
    float radius = jet_shape.x + jet_shape.y * along * jet_shape.z;

    vec3 axis = normalize(body_axis.xyz);
    vec3 right = normalize(body_right.xyz);
    vec3 forward = cross(axis, right);

    vec3 local = (right * p3d_Vertex.x + forward * p3d_Vertex.z) * radius
               + axis * (along * jet_shape.z);

    gl_Position = p3d_ModelViewProjectionMatrix
                * vec4(compress(body_offset.xyz + local), 1.0);

    v_uv = p3d_MultiTexCoord0;
}
""".replace("__ORIENTED__", _ORIENTED_BODY)

JET_FRAGMENT = """
#version 150

uniform vec4 jet_colour;

// x: brightness, y: time, z: length in model units
uniform vec4 jet_params;

in vec2 v_uv;

out vec4 frag_colour;

__TONEMAP__

void main() {
    // u runs along the jet from base to tip, v around it.
    float along = clamp(v_uv.x, 0.0, 1.0);
    float around = v_uv.y;

    // The jet is collimated near the hole and spreads as it goes,
    // so it dims along its length.
    float falloff = exp(-2.2 * along);

    // Knots of shocked plasma travelling outward at nearly the speed
    // of light, which is what a real jet is actually made of.
    float knots = 0.65 + 0.35 * sin((along * 18.0 - jet_params.y * 0.6) * 3.14159);

    // Brighter along the spine than at the sheath.
    float spine = pow(1.0 - abs(around * 2.0 - 1.0), 1.6);

    float intensity = jet_params.x * falloff * knots * spine;

    frag_colour = vec4(tonemap(jet_colour.rgb * intensity), 1.0);
}
""".replace("__TONEMAP__", _TONEMAP)


# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------

_cache = {}


def _make(name, vertex, fragment):
    if name not in _cache:
        _cache[name] = Shader(
            language=Shader.GLSL, vertex=vertex, fragment=fragment
        )

    return _cache[name]


SHADOW_FRAGMENT = """
#version 150

in vec2 v_uv;

out vec4 frag_colour;

void main() {
    if (dot(v_uv, v_uv) > 1.0) {
        discard;
    }

    frag_colour = vec4(0.0, 0.0, 0.0, 1.0);
}
"""


def shadow_shader():
    return _make("shadow", CORONA_VERTEX, SHADOW_FRAGMENT)


def star_shader():
    return _make("star", _SPHERE_VERTEX, STAR_FRAGMENT)


def planet_shader():
    return _make("planet", _SPHERE_VERTEX, PLANET_FRAGMENT)


def corona_shader():
    return _make("corona", CORONA_VERTEX, CORONA_FRAGMENT)


def disc_shader():
    return _make("disc", DISC_VERTEX, DISC_FRAGMENT)


def jet_shader():
    return _make("jet", JET_VERTEX, JET_FRAGMENT)
