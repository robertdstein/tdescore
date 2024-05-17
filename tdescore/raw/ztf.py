"""
Module for downloading raw ZTF data
"""
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tdescore.paths import ampel_cache_dir
from tdescore.raw.augment import augment_alerts
from tdescore.raw.nuclear_sample import all_sources

logger = logging.getLogger(__name__)

try:
    from nuztf.ampel_api import ampel_api_lightcurve
except ImportError:
    logger.warning("nuztf not installed. Some functionality will be disabled.")
    ampel_api_lightcurve = None

try:
    from nuztf.ampel_api import ampel_api_alerts
except ImportError:
    ampel_api_alerts = None

OVERWRITE = False


def get_alert_path(source: str) -> Path:
    """
    Get path of json alert data for source

    :param source: source name
    :return: path
    """
    return ampel_cache_dir.joinpath(f"{source}.json")


def flatten_alert_data(alert_data):
    """
    Flatten alert data into a packet, deduplicating based on jd

    :param alert_data: List of alert data
    :return: A alert packet, with each previous candidate
    only appearing once in the prv_candidates list
    """
    # Go in reverse order to get the most recent alert first
    alerts = alert_data[0][::-1]

    base_alert = alerts[0]

    jds = [base_alert["candidate"]["jd"]]
    prv_candidates = []

    for prv_cand in base_alert["prv_candidates"]:
        jds.append(prv_cand["jd"])
        prv_candidates.append(prv_cand)

    for alert in alerts:
        if not alert["candidate"]["jd"] in jds:
            candidate = alert["candidate"]
            prv_candidates.append(candidate)
            jds.append(candidate["jd"])

        for prv_cand in alert["prv_candidates"]:
            if not prv_cand["jd"] in jds:
                prv_candidates.append(prv_cand)
                jds.append(prv_cand["jd"])

    base_alert["prv_candidates"] = prv_candidates
    return [base_alert]


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
            if ampel_api_alerts is None:
                raise ImportError("nuztf not installed. Cannot download data.")

            # query_res = ampel_api_alerts(
            #     ztf_name=source,
            # )
            # query_res = flatten_alert_data(query_res)

            query_res = ampel_api_lightcurve(
                ztf_name=source,
            )

            if query_res[0] is not None:
                if "detail" not in query_res[0].keys():
                    alert_data = augment_alerts(query_res[0])
                    with open(output_path, "w", encoding="utf8") as out_f:
                        out_f.write(json.dumps([alert_data]))

                    passed.append(source)
                else:
                    logger.error(f"No alert data for {source}")

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
