"""
Module for parsing cached data, and copying a subset of it
"""
import json

import numpy as np

from tdescore.download import panstarrs_path, ps_base_keys, ps_copy_keys, ps_filters


def parse_ps1(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = panstarrs_path(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            all_cache_data = json.load(cache_f)

        res = {}

        for key in ps_copy_keys:
            if key in all_cache_data.keys():
                res[key] = all_cache_data[key]

        for i, filter_1 in enumerate(ps_filters[:-1]):
            filter_2 = ps_filters[i + 1]
            for key in ps_base_keys:
                label = f"{filter_1}-{filter_2}_{key}"
                try:
                    m1 = all_cache_data[filter_1 + key]
                    m2 = all_cache_data[filter_2 + key]
                    col = m1 - m2
                except (TypeError, KeyError):
                    col = np.nan
                res[label] = col

        for f in ps_filters:
            try:
                diff = (
                    all_cache_data[f"{f}MeanPSFMag"] - all_cache_data[f"{f}MeanKronMag"]
                )
            except (TypeError, KeyError):
                diff = np.nan
            res[f"kron_m_psf_{f}"] = diff
    else:
        res = {}

    return res
