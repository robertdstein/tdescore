"""
Module for handling fitting color evolution for a lightcurve
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor


def linear_color(time, c_grad, c_intercept):
    """
    Model for a g-r color which changes linearly with time

    :param time: time
    :param c_grad: color gradient
    :param c_intercept: color at t=0
    :return: predicted g-r color
    """
    return c_grad * time.flatten() + c_intercept


def fit_second_band(
    lc_2: pd.DataFrame, gp_1: GaussianProcessRegressor
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to take a Gaussian Process model of a one-band lightcurve, and fit
    a linear color evolution which maps this to a second lightcurve band.

    :param lc_2: second band lightcurve data
    :param gp_1: gaussian lightcurve trained on one band
    :return: r_band lightcurve prediction function, optimal color parameters,
        fitted covariance
    """

    t_data = lc_2["time"].to_numpy().reshape(-1, 1)

    pred_g, sigma_g = gp_1.predict(t_data, return_std=True)

    def predicted_r_lightcurve(time, c_grad, c_intercept):
        pred_r = pred_g.flatten() + linear_color(time, c_grad, c_intercept)
        return pred_r

    # pylint: disable=W0632
    popt, pcov = curve_fit(
        predicted_r_lightcurve,
        t_data,
        lc_2["magpsf"].to_numpy(),
        sigma=sigma_g,
        bounds=((-1.0, -5.0), (1.0, 5.0)),
    )

    return popt, pcov
