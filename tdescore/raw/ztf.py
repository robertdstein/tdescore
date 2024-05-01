"""
Module for downloading raw ZTF data
"""
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tdescore.paths import ampel_cache_dir
from tdescore.raw.nuclear_sample import all_sources

logger = logging.getLogger(__name__)

try:
    from nuztf.ampel_api import ampel_api_lightcurve
except ImportError:
    logger.warning("nuztf not installed. Some functionality will be disabled.")
    ampel_api_lightcurve = None

OVERWRITE = False


def get_alert_path(source: str) -> Path:
    """
    Get path of json alert data for source

    :param source: source name
    :return: path
    """
    return ampel_cache_dir.joinpath(f"{source}.json")


def download_alert_data(
    sources: list[str] = all_sources, overwrite: bool = OVERWRITE
) -> list[str]:
    """
    Function to download ZTF alert data via AMPEL
    (https://doi.org/10.1051/0004-6361/201935634) for all sources

    :return: None
    """

    logger.info(
        "Checking for availability of raw ZTF data. "
        "Will download from Ampel if missing."
    )

    passed = []

    for source in tqdm(sources, smoothing=0.8):
        output_path = get_alert_path(source)

        if np.logical_and(output_path.exists(), not overwrite):
            passed.append(source)

        else:
            if ampel_api_lightcurve is None:
                raise ImportError("nuztf not installed. Cannot download data.")

            query_res = ampel_api_lightcurve(
                ztf_name=source,
            )

            if query_res[0] is not None:
                with open(output_path, "w", encoding="utf8") as out_f:
                    out_f.write(json.dumps(query_res))

                passed.append(source)

    return passed


# def convert_pickle(sources: list[str] = all_sources):
#     """
#     Convert old ampel data from pickle to json (aka 'safe-ify code')
#
#     :param sources: list of sources
#     :return: None
#     """
#
#     for source in tqdm(sources):
#         old_path = get_old_alert_path(source)
#
#         with open(old_path, "rb") as alert_file:
#             query_res = pickle.load(alert_file)
#
#         new_path = get_alert_path(source)
#
#         with open(new_path, "w", encoding="utf8") as out_f:
#             out_f.write(json.dumps(query_res))
