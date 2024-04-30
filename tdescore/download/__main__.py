"""
Module to bulk download crossmatch data
"""
import logging

from tdescore.download.all import download_all
from tdescore.raw import load_raw_sources

logging.getLogger().setLevel(logging.INFO)


raw_source_table = load_raw_sources()
download_all(raw_source_table)
