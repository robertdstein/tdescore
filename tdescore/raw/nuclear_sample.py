"""
Module containing the starting list of all raw nuclear sources
"""
import pandas as pd

from tdescore.paths import data_dir
from tdescore.raw.tde import all_tdes

v2_sample_path = data_dir.joinpath("TransientTable.csv")
if v2_sample_path.exists():
    initial_sources = pd.read_csv(v2_sample_path)
    initial_sources["ztf_name"] = initial_sources["ztf_id"]
    for column in ["crossmatch_bts_class", "fritz_class"]:
        initial_sources[column] = None

else:
    initial_sample_path = data_dir.joinpath("nuclear_sample.csv")
    initial_sources = pd.read_csv(initial_sample_path)

missing = pd.DataFrame(
    [
        {"ztf_name": x}
        for x in all_tdes
        if x not in initial_sources["ztf_name"].to_list()
    ]
)
initial_sources = pd.concat([initial_sources, missing], ignore_index=True)
initial_sources.sort_values(by=["ztf_name"], inplace=True)

all_sources = initial_sources["ztf_name"].tolist()
