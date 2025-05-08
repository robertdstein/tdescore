"""
Script for calculating the offset from the average position of the source
"""
import numpy as np

import pandas as pd
from astropy.coordinates import SkyCoord


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
