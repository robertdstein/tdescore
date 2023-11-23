"""
Module for collating the data for the classifier
"""
import logging

import numpy as np
import pandas as pd

from tdescore.classifier.features import relevant_columns
from tdescore.combine.parse import load_metadata
from tdescore.raw.tde import is_tde

logger = logging.getLogger(__name__)


def get_all_sources() -> pd.DataFrame:
    """
    Get all classified sources in the sample, dropping unclassified sources

    :return: DataFrame of all classified sources
    """
    all_sources = load_metadata()
    all_sources["class"] = is_tde(all_sources["ztf_name"])

    # If sncosmo failed, the fit was bad! Set chi2 to max
    for key in ["sncosmo_chi2pdof", "sncosmo_chisq"]:
        mask = pd.isnull(all_sources[key])
        all_sources.loc[mask, key] = max(all_sources[key])

    return all_sources


def get_classified_sources() -> pd.DataFrame:
    """
    Get all classified sources in the sample, dropping unclassified sources

    :return: DataFrame of all classified sources
    """
    combined_metadata = get_all_sources()
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
        columns = relevant_columns
    res = df[columns].to_numpy()
    return res


def get_train_data(columns: list[str] | None = None) -> np.ndarray:
    """
    Get the training data for the classifier

    :param columns: Columns to use
    :return: Numpy array of data
    """
    if columns is None:
        columns = relevant_columns
    logger.info(f"Using {len(relevant_columns)} features")
    classified_sources = get_classified_sources()
    return convert_to_train_dataset(classified_sources, columns=columns)
