"""
Module for parsing a crossmatch of ZTF sources to WISE and fritz
"""
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from tdescore.raw.nuclear_sample import initial_sources as all_sources


def get_classification(source: str) -> str | None:
    """
    Returns the classification for a ZTF source, if known

    :param source: source name
    :return: classification
    """
    match = all_sources["fritz_class"][all_sources["ztf_name"] == source]
    return match.to_numpy()[0] if len(match) > 0 else None


def get_crossmatch(source: str) -> pd.DataFrame:
    """
    Returns the crossmatch for a ZTF source, if known

    :param source: source name
    :return: classification
    """
    match = all_sources[all_sources["ztf_name"] == source]
    return match


def crossmatch_with_catalog(
    catalog: pd.DataFrame, ra_deg: float, dec_deg: float, rad_arcsec: float = 1.5
) -> pd.DataFrame | None:
    """
    Function to search a catalog for crossmatches

    :param catalog: catalog to check
    :param ra_deg: source ra
    :param dec_deg: source dec
    :param rad_arcsec: search radius
    :return: match, if it exists, or None
    """
    rad_deg = rad_arcsec / 60.0 / 60.0

    match = None

    dec_mask = np.logical_and(
        catalog["dec"] > max(dec_deg - rad_deg, -90.0),
        catalog["dec"] < min(dec_deg + rad_deg, 90.0),
    )

    if np.sum(dec_mask) > 0:
        cut = catalog[dec_mask]

        src = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

        s_cat = SkyCoord(ra=cut["ra"], dec=cut["dec"], unit=u.degree)

        idx, d2d, _ = src.match_to_catalog_sky(s_cat)

        if d2d.to(u.arcsec).value < rad_arcsec:
            match = cut.iloc[idx]

    return match
