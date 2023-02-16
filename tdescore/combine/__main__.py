"""
Main module, used to manually re-parse metadata
"""
import logging

from tdescore.combine.parse import combine_all_sources

# from tdescore.metadata.parse import parse_metadata
from tdescore.combine.plot import batch_plot_variables
from tdescore.raw import load_raw_sources

# parse_metadata()
# batch_plot_variables()


logging.getLogger().setLevel(logging.INFO)

combine_all_sources(load_raw_sources())
batch_plot_variables()
