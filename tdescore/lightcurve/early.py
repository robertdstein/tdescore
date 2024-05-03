"""
Module for analysing early lightcurve data
"""
import json
from pathlib import Path

import numpy as np

from tdescore.alerts import load_source_raw
from tdescore.paths import lightcurve_early_dir

TIME_KEY = "mjd"

ALERT_COPY_KEYS = [
    "rb",
    "distnr",
    "magdiff",
    "sigmapsf",
    "chipsf",
    "sumrat",
    "fwhm",
    "elong",
    "classtar",
    "chinr",
    "sky",
    "sharpnr",
    "scorr",
    # "magpsf",
]


def get_early_lightcurve_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_early_dir.joinpath(f"{source}.json")


def analyse_source_early_data(source: str):
    """
    Perform a lightcurve analysis on first detections

    :param source: ZTF source to analyse
    :return: None
    """

    all_alert_data = load_source_raw(source)

    first_det_time = all_alert_data[TIME_KEY].min()

    mask = all_alert_data[TIME_KEY] < (first_det_time + 1.0)

    early_alert_data = all_alert_data[mask]

    new_values = {}

    for key in ALERT_COPY_KEYS:
        val = np.nanmedian(early_alert_data[key])
        new_values[f"early_{key}"] = val

    output_path = get_early_lightcurve_path(source)
    with open(output_path, "w", encoding="utf8") as out_f:
        out_f.write(json.dumps(new_values))
