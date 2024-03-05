"""
Module for collating the data for the classifier
"""
import logging

import numpy as np
import pandas as pd

from tdescore.classifier.assign import assign_classification_origin
from tdescore.classifier.features import default_columns
from tdescore.combine.parse import load_metadata
from tdescore.raw.tde import is_tde

logger = logging.getLogger(__name__)


def get_all_sources(include_unattributed: bool = True) -> pd.DataFrame:
    """
    Get all classified sources in the sample, dropping unclassified sources

    :param include_unattributed: Whether to include unattributed sources
    :return: DataFrame of all classified sources
    """
    all_sources = load_metadata()
    all_sources["class"] = is_tde(all_sources["ztf_name"])

    all_sources["fritz_class"].replace([None, "None", "-", ""], None, inplace=True)
    all_sources["bts_class"].replace([None, "None", "-", ""], None, inplace=True)
    all_sources["growth_class"].replace([None, "None", "-", ""], None, inplace=True)
    all_sources["crossmatch_bts_class"].replace(
        [None, "None", "-", np.nan, "nan", ""], None, inplace=True
    )

    # If sncosmo failed, the fit was bad! Set chi2 to max
    for key in ["sncosmo_chi2pdof", "sncosmo_chisq"]:
        mask = pd.isnull(all_sources[key])
        all_sources.loc[mask, key] = max(all_sources[key])

    all_sources = assign_classification_origin(
        all_sources, non_spectra_marshal_classes=include_unattributed
    )

    return all_sources


def get_classified_sources(include_unattributed: bool = False) -> pd.DataFrame:
    """
    Get all classified sources in the sample, dropping unclassified sources

    :return: DataFrame of all classified sources
    """
    combined_metadata = get_all_sources(include_unattributed=include_unattributed)
    unclassified_mask = np.logical_and(
        ~combined_metadata["class"].to_numpy(),
        (
            pd.isnull(combined_metadata["fritz_class"])
            & pd.isnull(combined_metadata["crossmatch_bts_class"])
        ),
    )

    logger.info(
        f"Sample has {len(combined_metadata)} sources. "
        f"Dropping {np.sum(unclassified_mask)} unclassified sources, "
        f"leaving {np.sum(~unclassified_mask)} sources, of which "
        f"{np.sum(combined_metadata['class'])} are tdes"
    )
    classified_sources = (combined_metadata[~unclassified_mask]).sort_values(
        by=["ztf_name"]
    )

    classified_sources = assign_classification_origin(classified_sources)

    if not include_unattributed:
        classified_sources["subclass"].replace(["duplicate"], None, inplace=True)
        classified_sources = classified_sources[
            ~pd.isnull(classified_sources["subclass"])
        ]
        classified_sources.reset_index(drop=True, inplace=True)

    return classified_sources


def convert_to_train_dataset(
    df: pd.DataFrame, columns: list[str] | None = None
) -> np.ndarray:
    """
    Convert a dataframe to a numpy array, using only the relevant columns.
    This array can be used as input to a classifier

    :param df: Dataframe to convert
    :param columns: Columns to use
    :return: Numpy array of data
    """
    if columns is None:
        columns = default_columns
    res = df[columns].to_numpy()
    return res


def get_train_data(columns: list[str] | None = None) -> np.ndarray:
    """
    Get the training data for the classifier

    :param columns: Columns to use
    :return: Numpy array of data
    """
    if columns is None:
        columns = default_columns
    logger.info(f"Using {len(default_columns)} features")
    classified_sources = get_classified_sources()
    return convert_to_train_dataset(classified_sources, columns=columns)
