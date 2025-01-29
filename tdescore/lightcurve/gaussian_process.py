"""
Module for fitting data with gaussian processes
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tdescore.lightcurve.color import fit_second_band, linear_color
from tdescore.lightcurve.errors import InsufficientDataError

MINIMUM_NOISE_MAGNITUDE = 0.1


def get_gp_model(times, magnitudes, alpha=0.01) -> GaussianProcessRegressor:
    """
    Returns a TDE-inspired 1D Gaussian Process model, trained on provided data.

    The model consists of a RBF kernel of variable amplitude,
    and a timescale of 10-500 days. This is added to a white noise kernel
    which accounts for systematic uncertainty in photometry.

    :param times: times of detection
    :param magnitudes: magnitudes of detections
    :param alpha: uncertainty
    :return: gaussian process model
    """
    # Instantiate a Gaussian Process model
    kernel = (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 10.0))
        * RBF(length_scale=50.0, length_scale_bounds=(10.0, 5.0e2))
    ) + WhiteKernel(noise_level=0.4, noise_level_bounds=(MINIMUM_NOISE_MAGNITUDE, 1.0))

    gp_model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=20, alpha=alpha
    )

    t_array = np.atleast_2d(times).T

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        gp_model.fit(t_array, magnitudes)

    return gp_model


def fit_two_band_lightcurve(
    lc_1: pd.DataFrame,
    lc_2: pd.DataFrame,
) -> tuple[GaussianProcessRegressor, pd.DataFrame, np.ndarray]:
    """

    :param lc_1: primary-band lightcurve data
    :param lc_2: secondary-band lightcurve data
    :return: Gaussian Process model, combined/corrected data, optimal color parameters
    """

    mask = np.logical_and(
        lc_2["time"] > min(lc_1["time"]), lc_2["time"] < max(lc_1["time"])
    )

    if np.sum(mask) == 0:
        raise InsufficientDataError("No contemporaneous g-band+r-band data")

    gp_1 = get_gp_model(
        lc_1["time"].to_numpy(dtype=float),
        lc_1["magpsf"].to_numpy(dtype=float),
        alpha=1.0e-5,
    )

    popt, _ = fit_second_band(lc_2=lc_2[mask], gp_1=gp_1)

    lc_2_corrected = lc_2.copy()
    lc_2_corrected["magpsf"] = lc_2["magpsf"] - linear_color(
        lc_2["time"].to_numpy(), *popt
    )

    lc_combined = pd.concat([lc_1, lc_2_corrected])

    gp_combined = get_gp_model(
        lc_combined["time"].to_numpy(dtype=float),
        lc_combined["magpsf"].to_numpy(dtype=float),
        alpha=1.0e-5,
    )

    return gp_combined, lc_combined, popt
