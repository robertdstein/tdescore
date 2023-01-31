"""
Module for plotting the result of lightcurve fits
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from tdescore.data import get_classification
from tdescore.lightcurve.color import linear_color
from tdescore.paths import lightcurve_dir


# pylint: disable=R0913,R0914
def plot_lightcurve_fit(
    source: str,
    gp_combined: GaussianProcessRegressor,
    lc_1: pd.DataFrame,
    lc_2: pd.DataFrame,
    mag_offset: float,
    popt: np.ndarray,
    txt: str,
):
    """
    Plot a lightcurve fit

    :param source: Source name
    :param gp_combined: Gaussian process model
    :param lc_1: Primary data band
    :param lc_2: secondary data band
    :param mag_offset: offset from lightcurve transformation
    :param popt: optimal color parameters
    :param txt: text to plot
    :return: None
    """
    classification = get_classification(source)

    out_dir = lightcurve_dir.joinpath(f"{str(classification).replace(' ', '_')}")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir.joinpath(f"{source}.png")

    title = f"{source} ({classification})"

    t_array = np.linspace(
        min(lc_1["time"].tolist() + lc_2["time"].tolist()),
        max(lc_1["time"].tolist() + lc_2["time"].tolist()),
        1000,
    )
    y_pred_raw, sigma = gp_combined.predict(t_array.reshape(-1, 1), return_std=True)
    y_pred = mag_offset - y_pred_raw

    plt.figure(figsize=(5, 6))
    plt.suptitle(title)
    ax1 = plt.subplot(311)
    plt.fill(
        np.concatenate([t_array, t_array[::-1]]),
        np.concatenate([y_pred - 1.0 * sigma, (y_pred + 1.0 * sigma)[::-1]]),
        alpha=0.3,
        fc="g",
        ec="None",
        label="95% confidence interval",
    )
    plt.plot(t_array, y_pred, linestyle=":", color="g")
    plt.scatter(lc_1["time"], mag_offset - lc_1["magpsf"], c="g")

    # y_peak = min(y_pred)

    # t_peak = x[y_pred == y_peak]
    # plt.axvline(t_peak)

    ax2 = plt.subplot(312, sharex=ax1)
    y_pred_2 = y_pred - linear_color(t_array, *popt)

    plt.fill(
        np.concatenate([t_array, t_array[::-1]]),
        np.concatenate([y_pred_2 - 1.0 * sigma, (y_pred_2 + 1.0 * sigma)[::-1]]),
        alpha=0.3,
        fc="r",
        ec="None",
        label="95% confidence interval",
    )
    plt.plot(t_array, y_pred_2, linestyle=":", color="r")
    plt.scatter(lc_2["time"], mag_offset - lc_2["magpsf"], c="r")

    plt.xlabel("Time since discovery [days]")

    ax1.set_ylabel(r"$m_{g}$")
    ax2.set_ylabel(r"$m_{r}$")

    plt.setp(ax1.get_xticklabels(), visible=False)

    for axis in [ax1, ax2]:
        axis.set_xlim(left=-5.0)
        axis.invert_yaxis()

    plt.subplot(313)
    plt.annotate(txt, xy=(0.0, 0.0))
    plt.axis("off")
    plt.subplots_adjust(hspace=0.0)

    plt.savefig(out_path)
    plt.close()
