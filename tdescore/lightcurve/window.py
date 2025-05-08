"""
Module with analysing subsets of lightcurve data
"""
import numpy as np
import pandas as pd
from tdescore.alerts import load_data_raw
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.offset import offset_from_average_position, sigma_offset

import logging

logger = logging.getLogger(__name__)

THERMAL_WINDOWS = [
    14.0,
    30.0,
    60.0,
    90.0,
    180.0,
    365.0,
    None
]

ALERT_COPY_KEYS = [
    "rb",
    "distnr",
    "magdiff",
    "sigmapsf",
    "chipsf",
    "sumrat",
    "fwhm",
    "elong",
    "chinr",
    "sky",
    "sharpnr",
    "scorr",
    "distpsnr1",
]

TIME_KEY = "mjd"


def get_window_data(
    source: str,
    window_days: float | None,
    include_fp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Select only detections from the first N days

    :param source: Name of source
    :param window_days: Window of days to consider
    :param include_fp: Include forced photometry
    :return: Detections, Limits, dictionary of parameters
    """
    all_photometry_data, all_limit_data = load_data_raw(source)

    all_alert_data = all_photometry_data[~all_photometry_data["fp_bool"]]

    mask = (all_alert_data["diffmaglim"] > 19.0) & (
        all_alert_data["isdiffpos"].isin(
            ["t", "T", "true", "True", True, 1, 1.0, "1", "1.0"]  # Thanks IPAC...
        )
    )

    if mask.sum() == 0:
        err = f"No good positive detections for {source}"
        logger.error(err)
        raise InsufficientDataError(err)

    all_alert_data = all_alert_data[mask]

    # Find Peak

    peak_data = all_alert_data[
        all_alert_data["magpsf"] == all_alert_data["magpsf"].min()
    ]
    peak_time = peak_data[TIME_KEY].values[0]

    pre_peak = all_alert_data[all_alert_data[TIME_KEY] <= peak_time]

    steps = pre_peak[TIME_KEY].diff()
    mask = steps > 90.0

    # If there are very early predetections, cut them off
    if mask.sum() > 0:
        idx_cut = np.where(mask)[-1][0]

        all_alert_data = all_alert_data.iloc[idx_cut:]

    first_det_time = all_alert_data[TIME_KEY].min()

    # Add in forced photometry if requested
    if include_fp:
        fp_mask = all_photometry_data["fp_bool"]
        all_alert_data = pd.concat([all_alert_data, all_photometry_data[fp_mask]])

    # Set window to end of data if not specified (i.e. include all data)
    if window_days is None:
        window_days = 0.001 + max(all_alert_data[TIME_KEY]) - first_det_time

    mask = all_alert_data[TIME_KEY] < (first_det_time + window_days)

    early_alert_data = all_alert_data[mask]

    if len(all_limit_data) > 0:
        early_limit_data = all_limit_data[
            all_limit_data[TIME_KEY] < (first_det_time + window_days)
        ]
    else:
        early_limit_data = pd.DataFrame(columns=early_alert_data.columns)

    age = max(all_alert_data[TIME_KEY]) - first_det_time

    return early_alert_data, early_limit_data, age

def analyse_window_data(
    source: str,
    window_days: float,
    label: str,
    include_fp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Perform a lightcurve analysis on first detections

    :param source: Name of source
    :param window_days: Window of days to consider
    :param label: Label for output
    :param include_fp: Include forced photometry
    :return: Detections, Limits, dictionary of parameters
    """
    early_alert_data, early_limit_data, age = get_window_data(
        source=source, window_days=window_days, include_fp=include_fp
    )

    new_values = {
        "age": age,
    }

    for key in ALERT_COPY_KEYS:
        val = np.nanmedian(early_alert_data[key])
        new_values[f"{label}_{key}"] = val

    try:
        offset_med = offset_from_average_position(early_alert_data)
        new_values[f"{label}_offset_med"] = offset_med
    except KeyError:
        pass

    try:
        offset_n_sigma, offset_ll, offset_ul = sigma_offset(early_alert_data)
        new_values[f"{label}_offset_n_sigma"] = offset_n_sigma
        new_values[f"{label}_offset_ll"] = offset_ll
        new_values[f"{label}_offset_ul"] = offset_ul
    except KeyError:
        pass

    new_values[f"{label}_n_detections"] = len(early_alert_data)

    return early_alert_data, early_limit_data, new_values