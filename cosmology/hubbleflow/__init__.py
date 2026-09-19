"""
Hubble flow: how comoving coordinates turn into proper distances,
recession velocities and observed redshifts.

The renderer stores every large-scale object in *comoving* coordinates
and multiplies by a(t) at draw time, which is exactly how the Hubble
flow works. This module supplies the observables that follow from that.
"""

import numpy as np

from simulation_constants import C, MPC


class HubbleFlow:
    def __init__(self, expansion):
        self.expansion = expansion

    def proper_distance(self, comoving_distance, cosmic_time):
        """
        Comoving separation -> proper separation at cosmic_time.
        """

        return np.asarray(comoving_distance) * self.expansion.scale_factor(cosmic_time)

    def recession_velocity(self, comoving_distance, cosmic_time):
        """
        v = H(t) * d_proper. Superluminal beyond the Hubble radius,
        which is physical: space itself is expanding.
        """

        d = self.proper_distance(comoving_distance, cosmic_time)

        return self.expansion.hubble_parameter(cosmic_time) * d

    def cosmological_redshift(self, emitted_time, observed_time):
        """
        Redshift of light emitted at one cosmic time and received at
        another: 1 + z = a(obs) / a(emit).
        """

        a_emit = self.expansion.scale_factor(emitted_time)
        a_obs = self.expansion.scale_factor(observed_time)

        return a_obs / a_emit - 1.0

    def doppler_factor(self, velocity):
        """
        Relativistic Doppler factor for a purely radial peculiar
        velocity, positive meaning recession.
        """

        beta = np.clip(np.asarray(velocity) / C, -0.999999, 0.999999)

        return np.sqrt((1.0 + beta) / (1.0 - beta))

    def hubble_radius_mpc(self, cosmic_time):
        return self.expansion.hubble_radius(cosmic_time) / MPC

    def is_receding_superluminally(self, comoving_distance, cosmic_time):
        return self.recession_velocity(comoving_distance, cosmic_time) > C
