"""
Module for parsing cached data, and copying a subset of it
"""
import json

from tdescore.download.catwise import catwise_path

CATWISE_COPY_KEYS = ["w1rchi2", "w1mpro", "w2mpro"]


def parse_catwise(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = catwise_path(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            all_cache_data = json.load(cache_f)

        res = {}

        for key in CATWISE_COPY_KEYS:
            if key in all_cache_data.keys():
                res[f"catwise_{key}"] = all_cache_data[key]

        try:
            res["catwise_w1_m_w2"] = res["catwise_w1mpro"] - res["catwise_w2mpro"]
        except (KeyError, TypeError):
            pass

    else:
        res = {}

    return res
