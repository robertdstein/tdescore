"""
Module with utility functions for the lightcurve analysis
"""
import numpy as np


def get_covariance_ellipse(
    popt: np.ndarray,
    pcov: np.ndarray,
    n_sigma: float = 1.0,
):
    """
    Get the covariance ellipse for a 2D Gaussian fit

    :param popt: Parameters of the Gaussian fit
    :param pcov: Covariance matrix of the Gaussian fit
    :param n_sigma: Number of sigmas to plot
    :return: X, Y coordinates of the ellipse
    """
    angle_ellipse = np.arctan2(pcov[0, 1], pcov[0, 0] - pcov[1, 1])

    angle = np.linspace(0.0, 2 * np.pi, 100)

    delta_x = n_sigma * np.sqrt(pcov[0, 0]) * np.cos(angle)
    delta_y = n_sigma * np.sqrt(pcov[1, 1]) * np.sin(angle)

    x_pos = popt[0] + delta_x * np.cos(angle_ellipse) - delta_y * np.sin(angle_ellipse)
    y_pos = popt[1] + delta_x * np.sin(angle_ellipse) + delta_y * np.cos(angle_ellipse)

    return x_pos, y_pos
