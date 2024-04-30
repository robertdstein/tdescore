"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download import ps1strm_path

STRM_COPY_KEYS = [
    "prob_Galaxy",
    "prob_Star",
    "prob_QSO",
    "z_phot",
    "z_phot0",
    "z_photErr",
    "extrapolation_Class",
    "extrapolation_Photoz",
]

STRM_CLASS_MAP = {
    "GALAXY": 0,
    "STAR": 1,
    "QSO": 2,
    "UNSURE": 3,
}


def parse_ps1strm(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = ps1strm_path(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            all_cache_data = json.load(cache_f)

        res = {}

        for key in STRM_COPY_KEYS:
            if key in all_cache_data.keys():
                res[f"strm_{key}"] = all_cache_data[key]

        if "class" in all_cache_data.keys():
            res["strm_class"] = STRM_CLASS_MAP[all_cache_data["class"]]

    else:
        res = {}

    return res
