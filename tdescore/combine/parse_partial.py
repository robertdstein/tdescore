"""
Module for parsing cached data, and copying a subset of it
"""
import json
from pathlib import Path
from typing import Callable

from tdescore.download import gaia_path
from tdescore.sncosmo.run_sncosmo import get_sncosmo_path


def parse_subset_of_cache(
    source_name: str, output_f: Callable[[str], Path], copy_keys: list[str]
) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :param output_f: path to json
    :param copy_keys: keys to copy over
    :return: pandas dataframe
    """

    cache_path = output_f(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            all_cache_data = json.load(cache_f)

        res = {}

        for key in copy_keys:
            if key in all_cache_data.keys():
                res[key] = all_cache_data[key]

    else:
        res = {}

    return res


catalog_tuples = [
    (
        gaia_path,
        [
            "parallax_over_error",
            "classprob_dsc_combmod_quasar",
            "classprob_dsc_combmod_galaxy",
            "classprob_dsc_combmod_star",
            "phot_variable_flag",
        ],
    ),
]


def parse_all_partial(source_name: str) -> dict:
    """
    Iteratively parse data from each cache for which a subset of data is required

    :param source_name: Name of source
    :return: dictionary containing only the requested keys from each cache
    """

    res = {}

    for path_f, copy_keys in catalog_tuples:
        res.update(
            parse_subset_of_cache(source_name, output_f=path_f, copy_keys=copy_keys)
        )

    return res
