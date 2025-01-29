"""
Module for calculating extinction from the
Schlegel, Finkbeiner, and Davis (1998) dust maps.
"""

import logging

import extinction
import numpy as np
import sfdmap
from astropy.coordinates import SkyCoord

from tdescore.paths import sfd_path

logger = logging.getLogger(__name__)

m = sfdmap.SFDMap(sfd_path.as_posix())


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
