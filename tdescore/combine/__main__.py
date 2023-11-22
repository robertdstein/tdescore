"""
Main module, used to manually re-parse metadata
"""
import logging

from tdescore.combine.parse import combine_all_sources
from tdescore.combine.plot import batch_plot_variables
from tdescore.raw import load_raw_sources

logging.getLogger().setLevel(logging.INFO)

combine_all_sources(load_raw_sources())
batch_plot_variables()
