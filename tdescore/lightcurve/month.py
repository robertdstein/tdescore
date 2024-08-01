"""
Module for analysing early lightcurve data
"""
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from tdescore.classifications.crossmatch import get_classification
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.infant import TIME_KEY, analyse_window_data
from tdescore.lightcurve.plot import FIG_HEIGHT, FIG_WIDTH
from tdescore.paths import lightcurve_dir, lightcurve_month_dir

logger = logging.getLogger(__name__)

DEFAULT_FILL_VALUE = np.nan


def get_month_lightcurve_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_month_dir.joinpath(f"{source}.json")


def joint_line(data, gradient, intercept, color):
    """
    Linear rise function

    :param data: 2d array (time, color)
    :param gradient: gradient
    :param intercept: intercept
    :param color: color offset
    :return: value
    """
    return gradient * data.T[0] + intercept + color * data.T[1]


# pylint: disable=R0913,R0914
def plot_linear_fit(
    source: str,
    alert_df: pd.DataFrame,
    best_fit: np.ndarray,
    base_output_dir: Path = lightcurve_dir,
):
    """
    Plot a lightcurve fit

    :param source: Source name
    :param alert_df: DataFrame of alerts
    :param best_fit: best fit parameters
    :param base_output_dir: output directory
    :return: None
    """
    classification = get_classification(source)

    out_dir = base_output_dir.joinpath(f"{str(classification).replace(' ', '_')}")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir.joinpath(f"{source}_linear.png")

    title = f"{source}"
    if classification is not None:
        title += f" ({classification})"

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.suptitle(title)

    ax1 = plt.subplot(1, 1, 1)

    timearray = np.linspace(0.0, 30.0, 10)

    for fid in [1, 2, 3]:
        mask = alert_df["fid"] == fid
        filter_df = alert_df[mask]
        y_pred = (
            -joint_line(
                np.array([timearray, (fid - 1) * np.ones_like(timearray)]).T, *best_fit
            )
            + alert_df["magpsf"].iloc[0]
        )  # Add the offset back in

        col = ["g", "r", "orange"][fid - 1]

        plt.plot(timearray, y_pred, linestyle=":", color=col)
        plt.scatter(filter_df["time"], filter_df["magpsf"], c=col)

    plt.xlabel("Time since discovery [days]")

    ax1.set_ylabel(r"$m$")

    ax1.set_xlim(left=-5.0)
    ax1.invert_yaxis()

    sns.despine()

    plt.subplots_adjust(hspace=0.0)

    plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)


def analyse_source_month_data(source: str, base_output_dir: Optional[Path] = None):
    """
    Perform a lightcurve analysis on first detections

    :param source: ZTF source to analyse
    :param base_output_dir: output directory for plots (default None)
    :return: None
    """

    window_days = 30.0
    label = "month"

    try:
        early_alert_data, _, new_values = analyse_window_data(
            source=source, window_days=window_days, label=label
        )
        early_alert_data["time"] = (
            early_alert_data[TIME_KEY] - early_alert_data[TIME_KEY].min()
        )
        early_alert_data["mag"] = (
            early_alert_data["magpsf"].iloc[0] - early_alert_data["magpsf"]
        )
        early_alert_data["color"] = early_alert_data["fid"] - 1

        if len(early_alert_data) > 2:
            best_fit, _ = curve_fit(
                joint_line,
                xdata=early_alert_data[["time", "color"]].to_numpy(),
                ydata=early_alert_data["mag"],
                sigma=early_alert_data["sigmapsf"].to_numpy(dtype=float),
                p0=[0.0, 0.0, 0.0],
            )

            if len(set(early_alert_data["fid"])) == 1:
                # Set colour to zero if only one filter
                old_best_fit = best_fit[2]
                old_intercept = best_fit[1]
                color = early_alert_data["color"].iloc[0]
                best_fit[1] = old_intercept + color * old_best_fit
                best_fit[2] = 0.0

        else:
            best_fit = [DEFAULT_FILL_VALUE, DEFAULT_FILL_VALUE, DEFAULT_FILL_VALUE]

        new_values["month_rise"] = best_fit[0]
        new_values["month_intercept"] = best_fit[1]
        new_values["month_color"] = best_fit[2]

        for key in ["rise", "intercept", "color"]:
            padded_val = new_values[f"month_{key}"]
            if pd.isnull(padded_val):
                padded_val = 0.0

            new_values[f"month_{key}_padded"] = padded_val

        residuals = (
            early_alert_data["mag"]
            - joint_line(early_alert_data[["time", "color"]].to_numpy(), *best_fit)
            / early_alert_data["sigmapsf"]
        )

        new_values["month_chi2"] = np.sum(residuals**2)
        mean_residuals = np.mean(residuals**2)
        new_values["mean_month_chi2"] = mean_residuals

        if pd.isnull(mean_residuals):
            new_values["mean_month_chi2_padded"] = 1.0
        else:
            new_values["mean_month_chi2_padded"] = mean_residuals

        if base_output_dir is not None:
            plot_linear_fit(source, early_alert_data, best_fit, base_output_dir)

        output_path = get_month_lightcurve_path(source)
        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(new_values))

    except InsufficientDataError:
        logger.debug(f"Insufficient data for {source}")
