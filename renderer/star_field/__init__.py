"""
The star field: the stars of the galaxy the observer is currently in,
drawn as one GPU point batch.

Positions are stored in parsecs relative to the host galaxy's centre.
That keeps float32 precision at about a thousandth of a parsec even at
the edge of a large galaxy, which no absolute coordinate system could
manage across cosmological distances.
"""

import numpy as np

from simulation_constants import PARSEC, C, YEAR

from renderer import PointCloud, LIGHT_YEAR

# One parsec, in light years.
PARSEC_IN_LIGHT_YEARS = PARSEC / LIGHT_YEAR


class StarField:
    """
    Binds a StellarPopulation to a PointCloud.

    Positions are written once. After that only the rows whose stars
    actually changed state are re-uploaded, which is why a quarter of
    a million evolving stars costs almost nothing per frame.
    """

    def __init__(self, population, parent=None, shader=None):
        self.population = population

        self.cloud = PointCloud(
            count=population.count,
            unit_in_light_years=PARSEC_IN_LIGHT_YEARS,
            parent=parent,
            shader=shader,
            size_gain_pixels=2.0,
            max_point_size=90.0,
            min_point_size=0.0,
            faint_cutoff=-7.5,
            # Light adds, so the core of a galaxy is the sum of tens
            # of thousands of overlapping point spread functions. At
            # unit brightness that saturates to a white disc and the
            # spiral structure disappears into it.
            brightness=0.45,
        )

        self.cloud.set_positions(population.position_pc)

        self.refresh_all()

    # --------------------------------------------------------

    def rebind(self, population):
        """
        Point at a different galaxy's stars without rebuilding any GPU
        objects.

        The buffer is a fixed size, so arriving somewhere new is a
        couple of memory copies rather than allocating and uploading a
        fresh mesh, which would stall the frame it happened on.
        """

        if population.count != self.population.count:
            raise ValueError(
                "star populations must be the same size to rebind; "
                f"got {population.count}, expected {self.population.count}"
            )

        self.population = population

        self.cloud.set_positions(population.position_pc)

        self.refresh_all()

    # --------------------------------------------------------

    def refresh_all(self):
        """
        Recompute colour and brightness for every star and upload the
        whole buffer. Used at startup and after a time jump, when most
        of the population has changed at once.
        """

        population = self.population

        self.cloud.data[:, 3:6] = population.colour
        self.cloud.data[:, 6] = population.alpha
        self.cloud.data[:, 7] = population.log_luminosity
        self.cloud.data[:, 8] = population.size_gain

        self.cloud.upload()

    def refresh_rows(self, indices):
        """
        Update only the stars that changed since the last frame.
        """

        if indices is None or len(indices) == 0:
            return

        population = self.population
        data = self.cloud.data

        data[indices, 3:6] = population.colour[indices]
        data[indices, 6] = population.alpha[indices]
        data[indices, 7] = population.log_luminosity[indices]
        data[indices, 8] = population.size_gain[indices]

        self.cloud.upload_rows(indices)

    def set_camera_parsecs(self, position_pc):
        self.cloud.set_camera(position_pc, scale_factor=1.0)

    def set_visible(self, value):
        self.cloud.visible = value

    def destroy(self):
        self.cloud.destroy()
