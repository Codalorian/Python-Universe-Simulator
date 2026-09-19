"""
Physical and simulation constants.

SI units are used internally wherever practical.
Astronomical units are provided for convenience.
"""

import math

# ------------------------------------------------------------
# Fundamental constants
# ------------------------------------------------------------

C = 299_792_458.0
G = 6.67430e-11

K_B = 1.380649e-23
H_PLANCK = 6.62607015e-34
HBAR = H_PLANCK / (2.0 * math.pi)

SIGMA_SB = 5.670374419e-8

M_PROTON = 1.67262192369e-27
M_ELECTRON = 9.1093837015e-31

SIGMA_THOMSON = 6.6524587321e-29

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

DAY = 86_400.0

YEAR = 365.25 * DAY
KYR = 1.0e3 * YEAR
MYR = 1.0e6 * YEAR
GYR = 1.0e9 * YEAR

# ------------------------------------------------------------
# Astronomical distances
# ------------------------------------------------------------

AU = 149_597_870_700.0

PARSEC = 3.085677581491367e16
KPC = 1.0e3 * PARSEC
MPC = 1.0e6 * PARSEC
GPC = 1.0e9 * PARSEC

LIGHT_YEAR = C * YEAR

# ------------------------------------------------------------
# Masses
# ------------------------------------------------------------

EARTH_MASS = 5.9722e24
SUN_MASS = 1.98847e30

JUPITER_MASS = 1.89813e27

# ------------------------------------------------------------
# Radii / luminosities
# ------------------------------------------------------------

EARTH_RADIUS = 6.371e6
SUN_RADIUS = 6.957e8
JUPITER_RADIUS = 6.9911e7

SUN_LUMINOSITY = 3.828e26
SUN_TEMPERATURE = 5772.0

# ------------------------------------------------------------
# Cosmology (Planck-2018-like flat LambdaCDM)
# ------------------------------------------------------------

H0_KM_S_MPC = 67.66

H0 = H0_KM_S_MPC * 1000.0 / MPC

HUBBLE_TIME = 1.0 / H0

OMEGA_M = 0.3111
OMEGA_B = 0.04897

# Radiation density today (photons + 3 neutrino species).
OMEGA_R = 9.182e-5

# Dark energy is derived so the model is exactly flat, which keeps
# the Friedmann integration self-consistent.
OMEGA_LAMBDA = 1.0 - OMEGA_M - OMEGA_R

OMEGA_K = 0.0

CMB_TEMPERATURE = 2.7255

SIGMA_8 = 0.8102
N_SPECTRAL = 0.9665

# Critical density today.
RHO_CRIT = 3.0 * H0 * H0 / (8.0 * math.pi * G)

# Age of the universe today, solved numerically at import time by
# cosmology.expansion.scale_factor. This literal is the standard
# Planck-2018 value and is used only to bootstrap that solver.
UNIVERSE_PRESENT_AGE = 13.787 * GYR

# Landmark epochs (cosmic time since the Big Bang).
T_PLANCK_EPOCH = 5.39e-44
T_INFLATION_END = 1.0e-32
T_QUARK_HADRON = 1.0e-5
T_NUCLEOSYNTHESIS = 180.0
T_MATTER_RADIATION_EQ = 51_100.0 * YEAR
T_RECOMBINATION = 372_000.0 * YEAR
T_FIRST_STARS = 180.0 * MYR
T_REIONIZATION_END = 1.0 * GYR

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

RANDOM_SEED = 20_260_918

# Fixed physics timestep, in real seconds.
PHYSICS_TICK = 1.0 / 60.0

# Simulated years per real second, at rate index 0.
DEFAULT_TIME_RATE = 1.0e6

MAX_TIME_RATE = 1.0e12

# How far past the present day the user may travel.
MAX_FUTURE_TIME = UNIVERSE_PRESENT_AGE + 1.0 * GYR

# Population sizes. These are rendered as GPU point batches, so the
# cost is dominated by the vectorised physics pass, not by draw calls.
GALAXY_COUNT = 60_000
STAR_COUNT = 240_000

# Radius of the simulated comoving volume, in Mpc.
UNIVERSE_RADIUS_MPC = 6_000.0

# Radius of the detailed "home" galaxy, in kpc.
HOME_GALAXY_RADIUS_KPC = 26.0

# ------------------------------------------------------------
# Stellar physics
# ------------------------------------------------------------

MIN_STAR_MASS_SOLAR = 0.08
MAX_STAR_MASS_SOLAR = 150.0

# Approximate mass threshold for core-collapse supernovae.
CORE_COLLAPSE_MIN_MASS = 8.0

# Above this the collapse is direct, with no bright explosion.
DIRECT_COLLAPSE_MIN_MASS = 40.0

# Approximate pair-instability range.
PAIR_INSTABILITY_MIN_MASS = 140.0
PAIR_INSTABILITY_MAX_MASS = 260.0

# Chandrasekhar and Tolman-Oppenheimer-Volkoff limits, in Msun.
CHANDRASEKHAR_MASS = 1.44
TOV_MASS = 2.17

# Canonical core-collapse explosion energy (1 foe = 1e44 J).
FOE = 1.0e44
