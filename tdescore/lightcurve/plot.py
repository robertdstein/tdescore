"""
Module for plotting the result of lightcurve fits
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor

from tdescore.classifications.crossmatch import get_classification
from tdescore.lightcurve.color import linear_color
from tdescore.paths import lightcurve_dir

GOLDEN_RATIO = 1.618
INCH_TO_MM = 25.4
FIG_WIDTH = 160 / INCH_TO_MM
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO


# pylint: disable=R0913,R0914
def plot_lightcurve_fit(
    source: str,
    gp_combined: GaussianProcessRegressor,
    lc_1: pd.DataFrame,
    lc_2: pd.DataFrame,
    mag_offset: float,
    popt: np.ndarray,
    txt: Optional[str] = None,
    base_output_dir: Path = lightcurve_dir,
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
    :param base_output_dir: output directory
    :return: None
    """
    classification = get_classification(source)

    out_dir = base_output_dir.joinpath(f"{str(classification).replace(' ', '_')}")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir.joinpath(f"{source}.png")

    title = f"{source}"
    if classification is not None:
        title += f" ({classification})"

    t_array = np.linspace(
        min(lc_1["time"].tolist() + lc_2["time"].tolist()),
        max(lc_1["time"].tolist() + lc_2["time"].tolist()),
        1000,
    )
    y_pred_raw, sigma = gp_combined.predict(t_array.reshape(-1, 1), return_std=True)
    y_pred = mag_offset - y_pred_raw

    if txt is not None:
        n_ax = 3
    else:
        n_ax = 2

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT * n_ax / 2.0))
    plt.suptitle(title)

    ax1 = plt.subplot(n_ax, 1, 1)
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

    ax2 = plt.subplot(n_ax, 1, 2, sharex=ax1)
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

    sns.despine()

    if txt is not None:
        plt.subplot(313)
        plt.annotate(txt, xy=(0.0, 0.0))
        plt.axis("off")

    plt.subplots_adjust(hspace=0.0)

    plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)
