"""
Module for handling the raw source table
"""
import pandas as pd

from tdescore.paths import data_dir

raw_source_path = data_dir.joinpath("raw_sources.csv")


def load_raw_sources() -> pd.DataFrame:
    """
    Function to load raw source table

    :return: raw source table
    """
    return pd.read_csv(raw_source_path)
