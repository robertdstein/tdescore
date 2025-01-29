"""
Module for analysing early lightcurve data
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from tdescore.alerts import load_data_raw
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.paths import lightcurve_infant_dir

logger = logging.getLogger(__name__)

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
    "chinr",
    "sky",
    "sharpnr",
    "scorr",
    "distpsnr1",
]


def get_infant_lightcurve_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_infant_dir.joinpath(f"{source}.json")


def offset_from_average_position(alert_df: pd.DataFrame) -> float:
    """
    Get the offset from the median position of the source

    :param alert_df: DataFrame of alerts
    :return: Float offset from median
    """
    med_ra = np.nanmedian(alert_df["ra"])
    med_dec = np.nanmedian(alert_df["dec"])

    ps1_coords = SkyCoord(
        ra=alert_df.iloc[0]["ra_ps1"], dec=alert_df.iloc[0]["dec_ps1"], unit="deg"
    )

    dist = (
        ps1_coords.separation(SkyCoord(ra=med_ra, dec=med_dec, unit="deg"))
        .to("arcsec")
        .value
    )

    return dist


def sigma_offset(alert_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Get the offset from the median position of the source in arcsec and sigma

    :param alert_df: DataFrame of alerts
    :return: N sigma offset, 1 sig lower bound on offset, 1 sig upper bound on offset
    """

    if len(alert_df) == 1:
        return np.nan, np.nan, np.nan

    delta_ra = np.nanstd(alert_df["ra"])
    delta_dec = np.nanstd(alert_df["dec"])

    delta_rad = np.sqrt(delta_ra**2.0 + delta_dec**2.0) * 60.0 * 60.0

    med_ra = np.nanmedian(alert_df["ra"])
    med_dec = np.nanmedian(alert_df["dec"])

    ps1_coords = SkyCoord(
        ra=alert_df.iloc[0]["ra_ps1"], dec=alert_df.iloc[0]["dec_ps1"], unit="deg"
    )

    dist = (
        ps1_coords.separation(SkyCoord(ra=med_ra, dec=med_dec, unit="deg"))
        .to("arcsec")
        .value
    )

    offset_n_sigma = dist / delta_rad

    offset_ll = max([dist - delta_rad, 0])
    offset_ul = dist + delta_rad

    return offset_n_sigma, offset_ll, offset_ul


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

    mask = all_alert_data[TIME_KEY] < (first_det_time + window_days)

    early_alert_data = all_alert_data[mask]

    if len(all_limit_data) > 0:
        early_limit_data = all_limit_data[
            all_limit_data[TIME_KEY] < (first_det_time + window_days)
        ]
    else:
        early_limit_data = pd.DataFrame(columns=early_alert_data.columns)

    new_values = {
        "age": max(all_alert_data[TIME_KEY]) - first_det_time,
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
