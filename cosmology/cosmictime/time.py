"""
The central simulation clock.
"""

from dataclasses import dataclass

from simulation_constants import (
    YEAR,
    KYR,
    MYR,
    GYR,
)

from cosmology.expansion.scale_factor import PRESENT_AGE


@dataclass(slots=True)
class CosmicTime:
    """
    Cosmic time, stored as seconds since the Big Bang.

    The clock is monotone only when driven by advance(); jump_to() may
    move it in either direction so the user can revisit the Big Bang.
    """

    seconds: float = PRESENT_AGE

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cosmic time cannot advance by a negative interval.")

        self.seconds += seconds

    def jump_to(self, seconds: float) -> None:
        self.seconds = max(0.0, seconds)

    # Retained for callers that want the old name.
    def set_age(self, seconds: float) -> None:
        self.jump_to(seconds)

    @property
    def years(self) -> float:
        return self.seconds / YEAR

    @property
    def thousand_years(self) -> float:
        return self.seconds / KYR

    @property
    def million_years(self) -> float:
        return self.seconds / MYR

    @property
    def billion_years(self) -> float:
        return self.seconds / GYR

    def add_myr(self, value: float) -> None:
        self.advance(value * MYR)

    def add_gyr(self, value: float) -> None:
        self.advance(value * GYR)

    def __str__(self) -> str:
        return format_cosmic_time(self.seconds)


def format_cosmic_time(seconds: float) -> str:
    """
    Format a cosmic age, choosing units that stay readable across the
    sixty-odd orders of magnitude between the Planck epoch and today.
    """

    if seconds <= 0.0:
        return "t = 0"

    if seconds < 1.0e-6:
        return f"{seconds:.3e} s"

    if seconds < 60.0:
        return f"{seconds:.4g} s"

    if seconds < 3600.0:
        return f"{seconds / 60.0:.2f} minutes"

    if seconds < YEAR:
        return f"{seconds / 3600.0:.2f} hours"

    if seconds < KYR:
        return f"{seconds / YEAR:,.0f} years"

    if seconds < MYR:
        return f"{seconds / KYR:.1f} thousand years"

    if seconds < GYR:
        return f"{seconds / MYR:.2f} million years"

    return f"{seconds / GYR:.4f} billion years"
