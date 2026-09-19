"""
Procedural meshes for resolved bodies.

Each entity needs its own mesh: an Ursina Mesh is a single scene
graph node, so handing the same one to several entities merely
reparents it and leaves all but the last with nothing to draw.
"""

import numpy as np

from ursina import Mesh


def sphere_model(rings=32, segments=64):
    """
    A unit-diameter UV sphere.

    Ursina's stock sphere is under a thousand triangles, which is
    plenty for something a few pixels across and visibly polygonal
    once a star fills the screen.
    """

    theta = np.linspace(0.0, np.pi, rings + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)

    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]

    x = sin_t * np.cos(phi)[None, :]
    y = cos_t * np.ones_like(phi)[None, :]
    z = sin_t * np.sin(phi)[None, :]

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3) * 0.5

    row = segments + 1
    a = np.arange(rings)[:, None] * row + np.arange(segments)[None, :]

    triangles = np.stack([a, a + row, a + row + 1, a, a + row + 1, a + 1], axis=-1)

    return Mesh(
        vertices=vertices.astype(np.float32).ravel(),
        triangles=triangles.astype(np.uint32).ravel(),
        mode="triangle",
        static=True,
    )


def annulus_model(inner_fraction=0.08, rings=48, segments=192):
    """
    A flat ring in the XZ plane with unit outer radius: an accretion
    disc.

    The u texture coordinate is the fractional radius, so the shader
    can read the disc's temperature straight off a radial ramp, and v
    is the angle around, which is what the Doppler beaming needs.
    """

    # Geometric in radius, not uniform. Everything that matters
    # about a disc - the temperature peak, the orbital speeds, and
    # above all the lensing - is concentrated within a few times the
    # inner edge. Uniform rings put almost every vertex out in the
    # cold outer disc and leave the inner region so coarse that the
    # deflected mesh shows its own facets.
    radius = np.geomspace(inner_fraction, 1.0, rings + 1)

    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)

    r = radius[:, None]
    p = phi[None, :]

    x = r * np.cos(p)
    z = r * np.sin(p)
    y = np.zeros_like(x)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    uvs = np.stack(
        [
            np.broadcast_to(radius[:, None], x.shape),
            np.broadcast_to(phi[None, :] / (2.0 * np.pi), x.shape),
        ],
        axis=-1,
    ).reshape(-1, 2)

    row = segments + 1
    a = np.arange(rings)[:, None] * row + np.arange(segments)[None, :]

    triangles = np.stack([a, a + row, a + row + 1, a, a + row + 1, a + 1], axis=-1)

    return Mesh(
        vertices=vertices.astype(np.float32).ravel(),
        triangles=triangles.astype(np.uint32).ravel(),
        uvs=uvs.astype(np.float32).ravel(),
        mode="triangle",
        static=True,
    )


def jet_model(segments=48, sections=64):
    """
    The skeleton of one relativistic jet: a tube along +Y whose
    profile the shader fills in.

    The vertices carry a *unit* radial direction rather than a
    radius, because a jet's width and its length are set by
    completely different physics. Collimation happens within tens of
    gravitational radii of the hole, while the length is set by how
    far the jet can drill through the surrounding medium - a ratio
    of ten billion for a stellar-mass hole. Baking a width into the
    mesh as a fraction of its length puts the observer inside the
    jet no matter where they stand.

    u runs from the base at the black hole to the tip, v around the
    circumference, so the shader can brighten the spine and march
    shock knots outward along it.
    """

    # Sampled logarithmically along its length. A jet spans ten
    # decades between the collimation region and its tip, so uniform
    # sections leave the base completely unresolved: the first one
    # alone would be wider than the observer's whole distance from
    # the hole, putting them inside the jet wherever they stood.
    t = np.concatenate(([0.0], np.geomspace(1.0e-7, 1.0, sections)))

    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)

    p = phi[None, :]

    x = np.broadcast_to(np.cos(p), (len(t), segments + 1))
    z = np.broadcast_to(np.sin(p), (len(t), segments + 1))
    y = np.broadcast_to(t[:, None], x.shape)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    uvs = np.stack(
        [
            np.broadcast_to(t[:, None], x.shape),
            np.broadcast_to(phi[None, :] / (2.0 * np.pi), x.shape),
        ],
        axis=-1,
    ).reshape(-1, 2)

    row = segments + 1
    a = np.arange(len(t) - 1)[:, None] * row + np.arange(segments)[None, :]

    triangles = np.stack([a, a + row, a + row + 1, a, a + row + 1, a + 1], axis=-1)

    return Mesh(
        vertices=vertices.astype(np.float32).ravel(),
        triangles=triangles.astype(np.uint32).ravel(),
        uvs=uvs.astype(np.float32).ravel(),
        mode="triangle",
        static=True,
    )


def billboard_model():
    """
    A unit quad centred on the origin, facing +Z: the glare around a
    bright source.
    """

    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    triangles = np.array([0, 1, 3, 0, 3, 2])

    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    return Mesh(
        vertices=vertices.astype(np.float32).ravel(),
        triangles=triangles.astype(np.uint32).ravel(),
        uvs=uvs.astype(np.float32).ravel(),
        mode="triangle",
        static=True,
    )


def ring_outline(segments=128):
    """
    A closed line loop of unit radius in the XY plane: an orbit path.
    """

    angle = np.linspace(0.0, 2.0 * np.pi, segments)

    vertices = np.stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
    )

    return Mesh(
        vertices=vertices.astype(np.float32).ravel(),
        mode="line",
        thickness=1.0,
        static=False,
    )
