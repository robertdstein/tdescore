"""
Module for parsing ZTF alert data
"""
import json
import logging

import numpy as np
import pandas as pd

from tdescore.raw.augment import CorruptedAlertError, alert_to_pandas
from tdescore.raw.ztf import download_alert_data, get_alert_path

logger = logging.getLogger(__name__)

lightcurve_columns = ["time", "magpsf", "sigmapsf"]


def load_data_raw(source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the alert data for a source, and convert it to a data frame

    :param source_name: ZTF name of source
    :return: dataframe of detections
    """
    path = get_alert_path(source_name)

    if not path.exists():
        download_alert_data(sources=[source_name])

    with open(path, "r", encoding="utf8") as alert_file:
        query_res = json.load(alert_file)

    if query_res[0] is None:
        path.unlink()
        err = f"Error: {source_name} has no alert data"
        logger.error(err)
        raise FileNotFoundError(err)

    try:
        source, limits = alert_to_pandas(query_res)
    except CorruptedAlertError as exc:
        path.unlink()
        err = f"Error: {source_name} has corrupted data"
        logger.error(err)
        raise FileNotFoundError(err) from exc

    source.sort_values(by=["mjd"], inplace=True)

    if "fp_bool" in source.columns:
        source["fp_bool"] = pd.notnull(source["fp_bool"]) | source["fp_bool"] == True
    else:
        source["fp_bool"] = False

    if limits is None:
        limits = pd.DataFrame()

    if len(limits) > 0:
        limits.sort_values(by=["mjd"], inplace=True)

    return source, limits


def load_source_raw(source_name: str) -> pd.DataFrame:
    """
    Load the alert data for a source, and convert it to a data frame

    :param source_name: ZTF name of source
    :return: dataframe of detections
    """
    source, _ = load_data_raw(source_name)
    return source


def get_positive_detection_mask(raw_data: pd.DataFrame) -> np.ndarray:
    """
    Returns a mask for whether a given ZTF alert is
    a positive (rather than negative) detection

    :param raw_data: raw alert data
    :return: boolean mask
    """
    mask = np.array([x in [1, "t", True, "1"] for x in raw_data["isdiffpos"]])
    return mask


def clean_source(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean alert data, retaining only clean detections

    :param raw_data: all raw alerts
    :return: 'clean' alerts
    """
    positive_det_mask = get_positive_detection_mask(raw_data)

    clean = raw_data.copy()[positive_det_mask]

    all_mask = np.ones(len(clean), dtype=bool)

    for _, mask in enumerate(
        [
            clean["nbad"] < 1,
            clean["fwhm"] < 5,
            # clean["elong"] < 1.3,
            # abs(clean["magdiff"]) < 0.3,
            clean["distnr"] < 1.0,
            clean["rb"] > 0.3,
            clean["diffmaglim"] > 19.0,
        ]
    ):
        all_mask *= mask

    clean = clean[all_mask]
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
    extinction_g: float,
    extinction_r: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Function to convert a full alert dataframe to two compressed lightcurve vectors.
    Each contains transformed variables used for analysis.

    :param full_alert_data: full dataframe of detections
    :param extinction_g: extinction in g-band
    :param extinction_r: extinction in r-band
    :return: r-band lightcurve, g-band lightcurve, magnitude offset
    """
    mask = np.logical_or(full_alert_data["fid"] == 1, full_alert_data["fid"] == 2)

    full_alert_data = full_alert_data[mask].copy()

    full_alert_data["time"] = full_alert_data["mjd"] - min(full_alert_data["mjd"])

    full_alert_data["magpsf"] = full_alert_data["magpsf"].astype(float)

    full_alert_data.loc[full_alert_data["fid"] == 1, ["magpsf"]] -= extinction_g
    full_alert_data.loc[full_alert_data["fid"] == 2, ["magpsf"]] -= extinction_r

    offset = max(full_alert_data["magpsf"])

    full_alert_data["magpsf"] = -full_alert_data["magpsf"] + offset

    lc_g = full_alert_data[full_alert_data["fid"] == 1]
    lc_r = full_alert_data[full_alert_data["fid"] == 2]

    return lc_g[lightcurve_columns], lc_r[lightcurve_columns], offset
