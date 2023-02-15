"""
Module for downloading raw ZTF data
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from nuztf.ampel_api import ampel_api_lightcurve
from tqdm import tqdm

from tdescore.paths import ampel_cache_dir, data_dir
from tdescore.raw.nuclear_sample import all_sources

OVERWRITE = False

logger = logging.getLogger(__name__)

old_ampel_cache_dir = data_dir.joinpath("ampel_old")


def get_alert_path(source: str) -> Path:
    """
    Get path of json alert data for source

    :param source: source name
    :return: path
    """
    return ampel_cache_dir.joinpath(f"{source}.json")


def get_old_alert_path(source: str) -> Path:
    """
    Get path of deprecated pickle data for source

    :param source: source name
    :return: path
    """
    return old_ampel_cache_dir.joinpath(f"{source}.pkl")


# def download_alert_data(sources: list[str]) -> None:
#     """
#     Function to download ZTF alert data via AMPEL
#     (https://doi.org/10.1051/0004-6361/201935634) for all sources
#
#     :return: None
#     """
#
#     for source in tqdm(sources, smoothing=0.8):
#
#         output_path = ampel_cache_dir.joinpath(f"{source}.pkl")
#
#         if np.logical_and(output_path.exists(), not OVERWRITE):
#             pass
#
#         else:
#
#             query_res = ampel_api_lightcurve(
#                 ztf_name=source,
#             )
#
#             with open(output_path, "wb") as alert_file:
#                 pickle.dump(query_res, alert_file)


def download_alert_data(sources: list[str] = all_sources) -> None:
    """
    Function to download ZTF alert data via AMPEL
    (https://doi.org/10.1051/0004-6361/201935634) for all sources

    :return: None
    """

    logger.info(
        "Checking for availability of raw ZTF data. "
        "Will download from Ampel if missing."
    )

    for source in tqdm(sources, smoothing=0.8):

        output_path = get_alert_path(source)

        if np.logical_and(output_path.exists(), not OVERWRITE):
            pass

        else:

            query_res = ampel_api_lightcurve(
                ztf_name=source,
            )

            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(query_res))


def convert_pickle(sources: list[str] = all_sources):
    """
    Convert old ampel data from pickle to json (aka 'safe-ify code')

    :param sources: list of sources
    :return: None
    """

    for source in tqdm(sources):

        old_path = get_old_alert_path(source)

        with open(old_path, "rb") as alert_file:
            query_res = pickle.load(alert_file)

        new_path = get_alert_path(source)

        with open(new_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(query_res))
