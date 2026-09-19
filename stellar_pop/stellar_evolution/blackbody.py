"""
Blackbody colour.

A star's apparent colour is the CIE tristimulus integral of its Planck
spectrum. Rather than fake it with a hand-tuned temperature ramp, this
module does the integral properly:

    1. Planck's law B_lambda(T) over 360-830 nm.
    2. Weight by the CIE 1931 2-degree colour matching functions, using
       the multi-lobe Gaussian fits of Wyman, Sloan & Shirley (2013).
    3. Convert XYZ to linear sRGB, normalise to constant luminance and
       desaturate any out-of-gamut result.

The integral is evaluated once at import for a log-spaced temperature
table, after which lookup is a vectorised interpolation.
"""

import numpy as np

from simulation_constants import H_PLANCK, C, K_B

_LAMBDA_MIN_NM = 360.0
_LAMBDA_MAX_NM = 830.0
_LAMBDA_STEPS = 471

_T_MIN = 500.0
_T_MAX = 120_000.0
_T_STEPS = 512


def _piecewise_gaussian(x, mu, sigma_low, sigma_high):
    sigma = np.where(x < mu, sigma_low, sigma_high)
    t = (x - mu) / sigma
    return np.exp(-0.5 * t * t)


def cie_matching_functions(lambda_nm):
    """
    CIE 1931 2-degree standard observer, Wyman et al. (2013) fits.
    """

    x = (
        1.056 * _piecewise_gaussian(lambda_nm, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gaussian(lambda_nm, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gaussian(lambda_nm, 501.1, 20.4, 26.2)
    )

    y = (
        0.821 * _piecewise_gaussian(lambda_nm, 568.8, 46.9, 40.5)
        + 0.286 * _piecewise_gaussian(lambda_nm, 530.9, 16.3, 31.1)
    )

    z = (
        1.217 * _piecewise_gaussian(lambda_nm, 437.0, 11.8, 36.0)
        + 0.681 * _piecewise_gaussian(lambda_nm, 459.0, 26.0, 13.8)
    )

    return x, y, z


def planck_spectral_radiance(lambda_m, temperature):
    """
    B_lambda(T) in W / (m^2 sr m). Shapes broadcast.
    """

    lam = np.asarray(lambda_m, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)

    a = 2.0 * H_PLANCK * C * C / lam**5
    b = H_PLANCK * C / (lam * K_B * t)

    # expm1 keeps the Rayleigh-Jeans limit accurate.
    return a / np.expm1(np.clip(b, 1.0e-12, 700.0))


# sRGB (D65) from CIE XYZ.
_XYZ_TO_RGB = np.array(
    [
        [3.2406255, -1.5372080, -0.4986286],
        [-0.9689307, 1.8757561, 0.0415175],
        [0.0557101, -0.2040211, 1.0569959],
    ]
)


def _build_colour_table():
    lam_nm = np.linspace(_LAMBDA_MIN_NM, _LAMBDA_MAX_NM, _LAMBDA_STEPS)
    lam_m = lam_nm * 1.0e-9

    cx, cy, cz = cie_matching_functions(lam_nm)

    temperatures = np.geomspace(_T_MIN, _T_MAX, _T_STEPS)

    # (T, lambda) grid of spectral radiance.
    radiance = planck_spectral_radiance(lam_m[None, :], temperatures[:, None])

    d_lambda = lam_m[1] - lam_m[0]

    big_x = (radiance * cx[None, :]).sum(axis=1) * d_lambda
    big_y = (radiance * cy[None, :]).sum(axis=1) * d_lambda
    big_z = (radiance * cz[None, :]).sum(axis=1) * d_lambda

    xyz = np.stack([big_x, big_y, big_z], axis=1)

    # Normalise to unit luminance so the table encodes hue, not
    # brightness. Brightness is applied separately by the renderer.
    xyz /= np.maximum(big_y[:, None], 1.0e-30)

    rgb = xyz @ _XYZ_TO_RGB.T

    # Desaturate towards white until nothing is negative, which is the
    # standard way to bring an out-of-gamut spectral colour into sRGB.
    deficit = np.maximum(0.0, -rgb.min(axis=1, keepdims=True))
    rgb = rgb + deficit

    rgb /= np.maximum(rgb.max(axis=1, keepdims=True), 1.0e-30)

    # Encode to sRGB.
    rgb = np.clip(rgb, 0.0, 1.0)

    rgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )

    return temperatures, np.clip(rgb, 0.0, 1.0)


_T_TABLE, _RGB_TABLE = _build_colour_table()

# The table is geometric in temperature, so a lookup is a logarithm
# and a multiply rather than a binary search. Converting a quarter of
# a million temperatures to colour is then a single gather, which is
# an order of magnitude faster than interpolating three channels and
# indistinguishable on screen at 512 steps across the range.
_RGB_TABLE_F32 = np.ascontiguousarray(_RGB_TABLE, dtype=np.float32)

_LOG_T_MIN = np.log(_T_MIN)
_LOG_T_SPAN = np.log(_T_MAX) - _LOG_T_MIN
_INDEX_SCALE = (_T_STEPS - 1) / _LOG_T_SPAN


def temperature_to_rgb(temperature):
    """
    Effective temperature (K) -> sRGB in 0..1.

    Accepts a scalar or an array; returns shape (..., 3) as float32.
    """

    t = np.clip(np.asarray(temperature, dtype=np.float64), _T_MIN, _T_MAX)

    index = ((np.log(t) - _LOG_T_MIN) * _INDEX_SCALE).astype(np.intp)

    return _RGB_TABLE_F32[index]


def temperature_to_rgb_exact(temperature):
    """
    The same conversion by interpolation rather than table lookup, for
    callers that care about the last fraction of a percent.
    """

    t = np.clip(np.asarray(temperature, dtype=np.float64), _T_MIN, _T_MAX)

    out = np.empty(t.shape + (3,), dtype=np.float32)

    for channel in range(3):
        out[..., channel] = np.interp(t, _T_TABLE, _RGB_TABLE[:, channel])

    return out


def wien_peak_nm(temperature):
    """
    Wavelength of peak emission, from Wien's displacement law.
    """

    return 2.897771955e-3 / np.asarray(temperature, dtype=np.float64) * 1.0e9


def spectral_class(temperature) -> str:
    """
    Harvard spectral classification from effective temperature.
    """

    t = float(temperature)

    if t >= 30_000:
        return "O"
    if t >= 10_000:
        return "B"
    if t >= 7_500:
        return "A"
    if t >= 6_000:
        return "F"
    if t >= 5_200:
        return "G"
    if t >= 3_700:
        return "K"
    if t >= 2_400:
        return "M"

    return "L"
