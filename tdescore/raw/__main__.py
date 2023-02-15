"""
Module for downloading and preprocessing raw ZTF data for nuclear sample
"""
import logging

from tdescore.raw.extract import combine_raw_source_data
from tdescore.raw.ztf import download_alert_data

logging.getLogger().setLevel(logging.INFO)

download_alert_data()
combine_raw_source_data()
