"""
Combine BOOM crossmatches from MilliQUAS
"""

import numpy as np
import pandas as pd
from blastwave import Source

COPY_KEYS = [
    "distance_arcsec",
]


def parse_milliquas_crossmatch(source: Source) -> dict:
    """
    Parse milliquas crossmatch data from a Source object.

    :param source: Source object containing milliquas crossmatch data
    :return: Source object with parsed milliquas crossmatch data
    """

    match_dict = {"has_milliquas": False}
    for key in COPY_KEYS:
        match_dict[f"milliquas_{key}"] = np.nan

    if source.crossmatches is not None:
        if "milliquas_v8" in source.crossmatches:
            if len(source.crossmatches["milliquas_v8"]) > 0:
                match = pd.Series(source.crossmatches["milliquas_v8"][0]).replace(
                    {None: np.nan}
                )
                match_dict["has_milliquas"] = True
                for key in COPY_KEYS:
                    match_dict[f"milliquas_{key}"] = match[key]

    return match_dict
