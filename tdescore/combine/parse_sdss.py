"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download import sdss_path


def parse_sdss(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = sdss_path(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            cat_res = json.load(cache_f)
    else:
        cat_res = {}

    res = {}

    for key in cat_res.keys():
        res[f"sdss_{key}"] = cat_res[key]

    return res
