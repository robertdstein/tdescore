"""
Module for parsing ZTF alert data
"""
import pickle

import pandas as pd
from nuztf.plot import alert_to_pandas

from tdescore.paths import ampel_cache_dir


def load_source_raw(source_name: str) -> pd.DataFrame:
    """
    Load the alert data for a source, and convert it to a data frame

    :param source_name: ZTF name of source
    :return: dataframe of detections
    """
    path = ampel_cache_dir.joinpath(f"{source_name}.pkl")
    with open(path, "rb") as alert_file:
        query_res = pickle.load(alert_file)
    source = alert_to_pandas(query_res)[0]
    source.sort_values(by=["mjd"], inplace=True)
    return source


def load_source_clean(source_name: str) -> pd.DataFrame:
    """
    Load the alert data for a source, and convert it to a data frame.
    Cleans the detections, to retain only 'good' detections.

    :param source_name: ZTF name of source
    :return: dataframe of clean detections
    """
    source = load_source_raw(source_name)

    positive_det_mask = [x in [1, "t", True] for x in source["isdiffpos"]]
    source = source[positive_det_mask]

    source = source[source["nbad"] < 1]
    source = source[source["fwhm"] < 5]
    source = source[source["elong"] < 1.3]

    source = source[abs(source["magdiff"]) < 0.3]

    source = source[source["distnr"] < 1]

    source = source[source["diffmaglim"] > 20.0]
    source = source[source["rb"] > 0.3]
    return source
