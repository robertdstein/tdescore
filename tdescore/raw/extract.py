"""
Module for extracting features for ML classification
"""
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from tdescore.alerts import clean_source, get_positive_detection_mask, load_source_raw
from tdescore.raw.nuclear_sample import initial_sources
from tdescore.raw.table import raw_source_path

logger = logging.getLogger(__name__)

ALERT_COPY_KEYS = [
    "sgscore1",
    "distpsnr1",
    "distnr",
    "chinr",
    "drb",
    "sharpnr",
    "magdiff",
    "classtar",
    "nneg",
    "sumrat",
    "ra",
    "dec",
]


def extract_alert_parameters(raw_alert_data: pd.DataFrame) -> dict:
    """
    Extract various metaparameters from raw alert data

    :param raw_alert_data: raw alert data
    :return: relevant metaparameters
    """
    param_dict = {}

    positive_mask = get_positive_detection_mask(raw_alert_data)
    positive_fraction = np.mean(positive_mask.astype(float))

    param_dict["positive_fraction"] = positive_fraction

    for ind in range(3):
        fid = ind + 1
        data = raw_alert_data[raw_alert_data["fid"] == fid]

        if len(data) > 0:
            det_peak = np.min(data["magpsf"])
        else:
            det_peak = np.nan

        param_dict[f"det_peak_{fid}"] = det_peak

    param_dict["det_peak_all"] = np.min(raw_alert_data["magpsf"])
    param_dict["raw_n_det"] = len(raw_alert_data)

    clean_data = clean_source(raw_alert_data)

    for key in ALERT_COPY_KEYS:
        val = np.nan

        try:
            non_nan = pd.notnull(clean_data[key])
            if np.sum(non_nan) > 0:
                val = np.nanmedian(clean_data[key])
            else:
                non_nan = pd.notnull(raw_alert_data[key])
                if np.sum(non_nan) > 0:
                    val = np.nanmedian(raw_alert_data[key])

        except KeyError:
            pass

        param_dict[key] = val

    return param_dict


def combine_raw_source_data(src_table: pd.DataFrame = initial_sources):
    """
    Combine raw source data for all ZTF objects

    :param src_table: Table of sources
    :return: None
    """

    logger.info("Extracting general parameters from ZTF alerts")

    new_table = pd.DataFrame()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        try:
            ztf_alerts = load_source_raw(row["ztf_name"])
            lightcurve_metadata = extract_alert_parameters(ztf_alerts)
            new = pd.Series(lightcurve_metadata)
            full = pd.concat([row, new])

            new_table = pd.concat([new_table, full], ignore_index=True, axis=1)
        except KeyError:
            logger.warning(f"Failed to load {row['ztf_name']}")

    new_table = new_table.transpose()

    with open(raw_source_path, "w", encoding="utf8") as output_f:
        new_table.to_json(output_f)
