"""
Module for analysing early lightcurve data
"""
import json
import logging
from pathlib import Path

import numpy as np

from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.paths import lightcurve_infant_dir
from tdescore.lightcurve.window import analyse_window_data, TIME_KEY


logger = logging.getLogger(__name__)


def get_infant_lightcurve_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_infant_dir.joinpath(f"{source}.json")


def analyse_source_early_data(source: str):
    """
    Perform a lightcurve analysis on first detections

    :param source: ZTF source to analyse
    :return: None
    """

    window_days = 1.5
    label = "infant"

    try:
        early_alert_data, early_limit_data, new_values = analyse_window_data(
            source=source, window_days=window_days, label=label
        )

        new_values["infant_has_g"] = bool((early_alert_data["fid"] == 1).any())
        new_values["infant_has_r"] = bool((early_alert_data["fid"] == 2).any())

        mask = early_limit_data["fid"] == early_alert_data["fid"].iloc[0]

        if mask.sum() > 0:
            last_upper_lim = early_limit_data[mask][TIME_KEY].max()

            delay = early_alert_data[TIME_KEY].min() - last_upper_lim
            rise = (
                early_limit_data[mask].iloc[-1]["diffmaglim"]
                - early_alert_data.iloc[0]["magpsf"]
            )

            grad = rise / delay

        else:
            delay = np.nan
            rise = np.nan
            grad = np.nan

        new_values["infant_ul_delay"] = delay
        new_values["infant_ul_rise"] = rise
        new_values["infant_ul_grad"] = grad

        output_path = get_infant_lightcurve_path(source)
        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(new_values))

    except InsufficientDataError:
        logger.debug(f"Insufficient data for {source}")
