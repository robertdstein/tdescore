"""
Module for calculating extinction from the
Schlegel, Finkbeiner, and Davis (1998) dust maps.
"""

import logging

import extinction
import numpy as np
import sfdmap
from astropy.coordinates import SkyCoord
import pandas as pd

from tdescore.paths import sfd_path

logger = logging.getLogger(__name__)

m = sfdmap.SFDMap(sfd_path.as_posix())

wavelengths = {
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
}

extra_wavelengths = {
    "UVW2": 2079.0,
    "U": 3465.0,
    "g": 4770.0,
    "J": 12350.0,
}


def get_extinction_correction(
    ra_deg: float,
    dec_deg: float,
    wavelengths: list[float] | None = None,
) -> float:
    """
    Apply extinction correction

    See ... citation
    """
    coordinates = SkyCoord(ra_deg, dec_deg, frame="icrs", unit="degree")
    ebv = m.ebv(coordinates)

    if wavelengths is None:
        wavelengths = [4770.0, 6231.0]

    wave = np.array(wavelengths)

    return extinction.fitzpatrick99(wave, 3.1 * ebv)


def apply_extinction_correction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply extinction correction to a DataFrame of candidates.

    :param df: DataFrame containing candidates with 'ra' and 'dec' columns
    :return: DataFrame with extinction correction applied
    """
    df = df.copy()
    df["filter"] = df["fid"].map({1: "g", 2: "r", 3: "i"})
    df["wavelength"] = df["filter"].map(wavelengths)

    ra = df["ra"].mean()
    dec = df["dec"].mean()

    for wavelength in df["wavelength"].unique():
        mask = df["wavelength"] == wavelength
        if mask.sum() > 0:
            ext = get_extinction_correction(
                ra_deg=ra, dec_deg=dec, wavelengths=[wavelength]
            )
            df.loc[mask, ["magpsf"]] -= ext

    return df
