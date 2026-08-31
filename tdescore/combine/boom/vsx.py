"""
Combine BOOM crossmatches from VSX
"""

import numpy as np
import pandas as pd
from blastwave import Source

COPY_KEYS = ["distance_arcsec", "var_flag"]


def parse_vsx_crossmatch(source: Source) -> dict:
    """
    Parse VSX crossmatch data from a Source object.

    :param source: Source object containing VSX crossmatch data
    :return: Source object with parsed VSX crossmatch data
    """

    match_dict = {}
    for key in COPY_KEYS:
        match_dict[f"VSX_{key}"] = np.nan

    if source.crossmatches is not None:
        if "VSX" in source.crossmatches:
            if len(source.crossmatches["VSX"]) > 0:
                match = pd.Series(source.crossmatches["VSX"][0]).replace({None: np.nan})
                for key in COPY_KEYS:
                    match_dict[f"VSX_{key}"] = match[key]

    return match_dict
