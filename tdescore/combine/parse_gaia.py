"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download.gaia import gaia_path

GAIA_COPY_KEYS = [
    "parallax_over_error",
    "in_qso_candidates",
    "in_galaxy_candidates",
    "phot_variable_flag",
    "classprob_dsc_combmod_quasar",
    "classprob_dsc_combmod_galaxy",
    "classprob_dsc_combmod_star",
]


def parse_gaia(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = gaia_path(source_name)

    with open(cache_path, "r", encoding="utf8") as cache_f:
        all_cache_data = json.load(cache_f)

    res = {}

    for key in GAIA_COPY_KEYS:
        if key in all_cache_data.keys():
            res[f"gaia_{key}"] = all_cache_data[key]

    try:
        aplx = abs(float(all_cache_data["parallax_over_error"]))
    except (TypeError, KeyError):
        aplx = 0.0

    res["gaia_aplx"] = aplx

    return res
