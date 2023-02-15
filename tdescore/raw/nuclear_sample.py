"""
Module containing the starting list of all raw nuclear sources
"""
import pandas as pd

from tdescore.paths import data_dir

initial_sample_path = data_dir.joinpath("nuclear_sample.csv")
all_sources = pd.read_csv(initial_sample_path)
all_sources.sort_values(by=["ztf_name"], inplace=True)
