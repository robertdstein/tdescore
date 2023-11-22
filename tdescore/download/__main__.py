"""
Module to bulk download crossmatch data
"""
import logging

from tdescore.download.gaia import download_gaia_data
from tdescore.download.mast import download_panstarrs_data
from tdescore.download.wise import download_wise_data
from tdescore.raw import load_raw_sources

logging.getLogger().setLevel(logging.INFO)

raw_source_table = load_raw_sources()

download_gaia_data(raw_source_table)
download_panstarrs_data(raw_source_table)
download_wise_data(raw_source_table)
