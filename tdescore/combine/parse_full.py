"""
Module for parsing data from caches, and copying all of it
"""
import json
from pathlib import Path
from typing import Callable

from tdescore.lightcurve.analyse import get_lightcurve_metadata_path
from tdescore.lightcurve.early import get_early_lightcurve_path


def parse_full(source_name: str, output_f: Callable[[str], Path]) -> dict:
    """
    Function to extract all available parameters from a source cache file

    :param source_name: Name of source
    :param output_f: path to json
    :return: pandas dataframe
    """

    cache_path = output_f(source_name)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            res = json.load(cache_f)

    else:
        res = {}

    return res


cache_fs = [get_early_lightcurve_path, get_lightcurve_metadata_path]


def parse_all_full(source_name: str) -> dict:
    """
    Iteratively loop over each cache for a source which needs to be parsed in full

    :param source_name: Name of source
    :return: dictionary of results
    """

    res = {}

    for path_f in cache_fs:
        res.update(parse_full(source_name, output_f=path_f))

    return res
