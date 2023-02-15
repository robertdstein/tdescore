"""
Module for downloading raw ZTF data
"""
import pickle

import numpy as np
from nuztf.ampel_api import ampel_api_lightcurve
from tqdm import tqdm

# from tdescore.data import all_sources
from tdescore.paths import ampel_cache_dir

OVERWRITE = False


def download_alert_data(sources: list[str]) -> None:
    """
    Function to download ZTF alert data via AMPEL
    (https://doi.org/10.1051/0004-6361/201935634) for all sources

    :return: None
    """

    # sources = list(all_sources["ztf_name"])

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


# if __name__ == "__main__":
#     download_alert_data()
