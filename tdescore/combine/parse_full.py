"""
Module for parsing data from caches, and copying all of it
"""
import json
from pathlib import Path
from typing import Callable
import logging

import numpy as np

from tdescore.lightcurve.analyse import get_lightcurve_metadata_path
from tdescore.lightcurve.gaussian_process import MINIMUM_NOISE_MAGNITUDE
from tdescore.lightcurve.infant import get_infant_lightcurve_path
from tdescore.lightcurve.month import get_month_lightcurve_path
from tdescore.lightcurve.week import get_week_lightcurve_path
from tdescore.download.legacy_survey import legacy_survey_path

logger = logging.getLogger(__name__)


def parse_full(source_name: str, output_f: Callable[[str], Path]) -> dict:
    """
    Function to extract all available parameters from a source cache file

    :param source_name: Name of source
    :param output_f: path to json
    :return: pandas dataframe
    """

    cache_path = output_f(source_name)

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf8") as cache_f:
                res = json.load(cache_f)
        except json.decoder.JSONDecodeError:
            logger.warning(f"Could not read cache file {cache_path}")
            cache_path.unlink(missing_ok=True)
            res = {}

    else:
        res = {}

    return res


cache_fs = [
    legacy_survey_path,
    get_infant_lightcurve_path,
    get_week_lightcurve_path,
    get_month_lightcurve_path,
    get_lightcurve_metadata_path,
]


def parse_all_full(source_name: str) -> dict:
    """
    Iteratively loop over each cache for a source which needs to be parsed in full

    :param source_name: Name of source
    :return: dictionary of results
    """

    res = {}

    for path_f in cache_fs:
        res.update(parse_full(source_name, output_f=path_f))

    try:
        res["high_noise"] = res["noise"] > MINIMUM_NOISE_MAGNITUDE + 0.01
    except KeyError:
        res["high_noise"] = np.nan

    return res
