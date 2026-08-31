"""
Combine BOOM crossmatches from CATWISE
"""

import numpy as np
import pandas as pd
from blastwave import Source

COPY_KEYS = [
    "distance_arcsec",
    "w1mpro",
    "w2mpro",
    "w1rchi2",
    "w2rchi2",
]


def parse_catwise_crossmatch(source: Source) -> dict:
    """
    Parse CATWISE crossmatch data from a Source object.

    :param source: Source object containing CATWISE crossmatch data
    :return: Source object with parsed CATWISE crossmatch data
    """

    match_dict = {
        "catwise_w1_m_w2": np.nan,
    }
    for key in COPY_KEYS:
        match_dict[f"catwise_{key}"] = np.nan

    if source.crossmatches is not None:
        if "CatWISE2020" in source.crossmatches:
            if len(source.crossmatches["CatWISE2020"]) > 0:
                match = pd.Series(source.crossmatches["CatWISE2020"][0]).replace(
                    {None: np.nan}
                )

                match_dict["catwise_w1_m_w2"] = match["w1mpro"] - match["w2mpro"]

                for key in COPY_KEYS:
                    match_dict[f"catwise_{key}"] = match[key]

    return match_dict
