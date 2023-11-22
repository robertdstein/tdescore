"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download.wise import wise_path

WISE_COPY_KEYS = ["w1_chi2", "w1mpro", "w2mpro", "w3mpro", "w4mpro"]


def parse_wise(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :param output_f: path to json
    :param copy_keys: keys to copy over
    :return: pandas dataframe
    """

    cache_path = wise_path(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            all_cache_data = json.load(cache_f)

        res = {}

        for key in WISE_COPY_KEYS:
            if key in all_cache_data.keys():
                res[key] = all_cache_data[key]

        try:
            res["w1_m_w2"] = res["w1mpro"] - res["w2mpro"]
            res["w3_m_w4"] = res["w3mpro"] - res["w4mpro"]
        except (KeyError, TypeError):
            pass

    else:
        res = {}

    return res
