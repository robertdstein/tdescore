"""
Module to bulk download crossmatch data
"""
import logging

from tdescore.download.fritz import download_fritz_data
from tdescore.download.gaia import download_gaia_data
from tdescore.download.mast import download_panstarrs_data
from tdescore.download.sdss import download_sdss_data
from tdescore.download.tns import download_tns_data
from tdescore.download.wise import download_wise_data
from tdescore.raw import load_raw_sources

logging.getLogger().setLevel(logging.INFO)

raw_source_table = load_raw_sources()

download_tns_data(raw_source_table)
download_gaia_data(raw_source_table)
download_panstarrs_data(raw_source_table)
download_wise_data(raw_source_table)
download_sdss_data(raw_source_table)
download_fritz_data(raw_source_table)
