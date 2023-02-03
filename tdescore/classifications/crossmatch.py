"""
Module for parsing a crossmatch of ZTF sources to WISE and fritz
"""
import numpy as np
import pandas as pd

from tdescore.paths import data_dir

crossmatch_path = data_dir.joinpath("candidate_crossmatch.csv")
all_sources = pd.read_csv(crossmatch_path)
all_sources.sort_values(by=["ztf_name"], inplace=True)

unclassified_mask = np.logical_and(
    pd.isnull(all_sources["crossmatch_bts_class"])
    | all_sources["crossmatch_bts_class"].transform(lambda x: x == "duplicate"),
    pd.isnull(all_sources["fritz_class"])
    | all_sources["fritz_class"].transform(lambda x: x == "duplicate"),
)

classified = all_sources[~unclassified_mask]

duplicate_mask = np.logical_or(
    classified["crossmatch_bts_class"].transform(lambda x: x == "duplicate"),
    classified["fritz_class"].transform(lambda x: x == "duplicate"),
)

classified = classified[~duplicate_mask]


def get_classification(source: str) -> str | None:
    """
    Returns the classification for a ZTF source, if known

    :param source: source name
    :return: classification
    """
    match = classified["fritz_class"][classified["ztf_name"] == source]
    return match.to_numpy()[0] if len(match) > 0 else None


def get_crossmatch(source: str) -> pd.DataFrame:
    """
    Returns the crossmatch for a ZTF source, if known

    :param source: source name
    :return: classification
    """
    match = all_sources[all_sources["ztf_name"] == source]
    return match
