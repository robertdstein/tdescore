"""
Module to download all crossmatch data for a table of sources
"""
import pandas as pd

from tdescore.download.fritz import download_fritz_data
from tdescore.download.gaia import download_gaia_data
from tdescore.download.kowalski import download_ps1strm_data
from tdescore.download.mast import download_panstarrs_data
from tdescore.download.sdss import download_sdss_data
from tdescore.download.tns import download_tns_data
from tdescore.download.wise import download_wise_data


def download_all(source_table: pd.DataFrame, include_optional: bool = True):
    """
    Function to download all crossmatch data for a table of sources

    :param source_table: Pandas DataFrame of sources
    :param include_optional: Download non-critical data not required for TDEScore
    :return: None
    """
    download_ps1strm_data(source_table)
    download_gaia_data(source_table)
    download_panstarrs_data(source_table)
    download_wise_data(source_table)
    if include_optional:
        download_tns_data(source_table)
        download_sdss_data(source_table)
        download_fritz_data(source_table)
