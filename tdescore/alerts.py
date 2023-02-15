"""
Module for parsing ZTF alert data
"""
import json

import numpy as np
import pandas as pd
from nuztf.plot import alert_to_pandas

from tdescore.paths import ampel_cache_dir
from tdescore.raw.ztf import download_alert_data

lightcurve_columns = ["time", "magpsf", "sigmapsf"]


def load_source_raw(source_name: str) -> pd.DataFrame:
    """
    Load the alert data for a source, and convert it to a data frame

    :param source_name: ZTF name of source
    :return: dataframe of detections
    """
    path = ampel_cache_dir.joinpath(f"{source_name}.pkl")

    if not path.exists():
        download_alert_data(sources=[source_name])

    with open(path, "r", encoding="utf8") as alert_file:
        query_res = json.load(alert_file)
    source, _ = alert_to_pandas(query_res)
    source.sort_values(by=["mjd"], inplace=True)
    return source


def get_positive_detection_mask(raw_data: pd.DataFrame) -> np.ndarray:
    """
    Returns a mask for whether a given ZTF alert is
    a positive (rather than negative) detection

    :param raw_data: raw alert data
    :return: boolean mask
    """
    return np.array([x in [1, "t", True] for x in raw_data["isdiffpos"]])


def clean_source(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean alert data, retaining only clean detections

    :param raw_data: all raw alerts
    :return: 'clean' alerts
    """
    positive_det_mask = get_positive_detection_mask(raw_data)
    clean = raw_data.copy()[positive_det_mask]

    clean = clean[clean["nbad"] < 1]
    clean = clean[clean["fwhm"] < 5]
    clean = clean[clean["elong"] < 1.3]

    clean = clean[abs(clean["magdiff"]) < 0.3]

    clean = clean[clean["distnr"] < 1]

    clean = clean[clean["diffmaglim"] > 20.0]
    clean = clean[clean["rb"] > 0.3]
    return clean


def load_source_clean(source_name: str) -> pd.DataFrame:
    """
    Load the alert data for a source, and convert it to a data frame.
    Cleans the detections, to retain only 'good' detections.

    :param source_name: ZTF name of source
    :return: dataframe of clean detections
    """
    source = clean_source(load_source_raw(source_name))
    return source


def get_lightcurve_vectors(
    full_alert_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Function to convert a full alert dataframe to two compressed lightcurve vectors.
    Each contains transformed variables used for analysis.

    :param full_alert_data: full dataframe of detections
    :return: r-band lightcurve, g-band lightcurve, magnitude offset
    """
    mask = np.logical_or(full_alert_data["fid"] == 1, full_alert_data["fid"] == 2)

    full_alert_data = full_alert_data[mask].copy()

    full_alert_data["time"] = full_alert_data["mjd"] - min(full_alert_data["mjd"])

    offset = max(full_alert_data["magpsf"])

    full_alert_data["magpsf"] = -full_alert_data["magpsf"] + offset

    lc_g = full_alert_data[full_alert_data["fid"] == 1]
    lc_r = full_alert_data[full_alert_data["fid"] == 2]

    return lc_g[lightcurve_columns], lc_r[lightcurve_columns], offset
