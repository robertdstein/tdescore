"""
Combine BOOM crossmatches from Gaia
"""

import numpy as np
import pandas as pd
from blastwave import Source

COPY_KEYS = ["distance_arcsec"]


def parse_gaia_crossmatch(source: Source) -> dict:
    """
    Parse Gaia crossmatch data from a Source object.

    :param source: Source object containing Gaia crossmatch data
    :return: Source object with parsed Gaia crossmatch data
    """

    match_dict = {"gaia_aplx": np.nan, "gaia_apmra": np.nan, "gaia_apmdec": np.nan}
    for key in COPY_KEYS:
        match_dict[f"gaia_{key}"] = np.nan

    if source.crossmatches is not None:
        if "Gaia_DR3" in source.crossmatches:
            if len(source.crossmatches["Gaia_DR3"]) > 0:
                match = pd.Series(source.crossmatches["Gaia_DR3"][0]).replace(
                    {None: np.nan}
                )

                match_dict["gaia_aplx"] = abs(
                    match["parallax"] / match["parallax_error"]
                )
                match_dict["gaia_apmra"] = abs(match["pmra"] / match["pmra_error"])
                match_dict["gaia_apmdec"] = abs(match["pmdec"] / match["pmdec_error"])

                for key in COPY_KEYS:
                    match_dict[f"gaia_{key}"] = match[key]

    return match_dict
