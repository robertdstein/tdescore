"""
Module for downloading, analysing and parsing data used for training classifier
"""
import logging
import pickle

import numpy as np
import pandas as pd
from nuztf.ampel_api import ampel_api_lightcurve
from tqdm import tqdm

from tdescore.paths import ampel_cache_dir, data_dir

logger = logging.getLogger(__name__)

OVERWRITE = False

ztf1_tde_list = pd.read_table(
    data_dir.joinpath("tde_names.txt"),
    skiprows=1,
    names=["AT Name", "ZTF Name", "Other1", "Other2", "Other3", "GOT Name", "spectype"],
)

crossmatch_path = data_dir.joinpath("candidate_crossmatch.csv")
all_sources = pd.read_csv(crossmatch_path)
all_sources.sort_values(by=["ztf_name"], inplace=True)

unclassified_mask = np.logical_and(
    pd.isnull(all_sources["crossmatch_bts_class"])
    | all_sources["crossmatch_bts_class"].transform(lambda x: x == "duplicate"),
    pd.isnull(all_sources["fritz_class"])
    | all_sources["fritz_class"].transform(lambda x: x == "duplicate"),
)

classified = all_sources[~unclassified_mask]

duplicate_mask = np.logical_or(
    classified["crossmatch_bts_class"].transform(lambda x: x == "duplicate"),
    classified["fritz_class"].transform(lambda x: x == "duplicate"),
)

classified = classified[~duplicate_mask]


def download_alert_data() -> None:
    """
    Function to download ZTF alert data via AMPEL () for all sources

    :return: None
    """
    sources = list(all_sources["ztf_name"])

    for source in tqdm(sources, smoothing=0.8):

        output_path = ampel_cache_dir.joinpath(f"{source}.pkl")

        if np.logical_and(output_path.exists(), not OVERWRITE):
            pass

        else:

            query_res = ampel_api_lightcurve(
                ztf_name=source,
            )

            with open(output_path, "wb") as alert_file:
                pickle.dump(query_res, alert_file)


if __name__ == "__main__":
    download_alert_data()
