"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download import tns_path

TNS_COPY_KEYS = ["redshift", "objname", "name_prefix", "dist_arcsec"]


def parse_tns(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """
    cache_path = tns_path(source_name)

    res = {}

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            cat_res = json.load(cache_f)

        if len(cat_res) > 0:
            for key in TNS_COPY_KEYS:
                res[f"tns_{key}"] = cat_res[key]

            res["tns_classification"] = cat_res["object_type"]["name"]

    return res
