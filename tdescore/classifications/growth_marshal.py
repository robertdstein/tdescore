"""
Module for performing growth crossmatching
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from tdescore.paths import data_dir

growth_spectra_path = data_dir.joinpath("growth_marshal_spectra.csv")
growth_class_path = data_dir.joinpath("growth_marshal_classes.csv")


def crossmatch_to_growth(src_data) -> pd.DataFrame:
    """
    Crossmatch to milliquas catalog

    :param src_data: array of all source ra values
    :return: updated_src_data
    """

    if not growth_spectra_path.exists():
        raise FileNotFoundError(f"No file found at {growth_spectra_path}")

    growth_spec_df = pd.read_csv(growth_spectra_path)
    counts = growth_spec_df["name"].value_counts()

    n_spec = []

    for _, row in tqdm(src_data.iterrows(), total=len(src_data)):
        if row["ztf_name"] in counts:
            n_spec.append(counts[row["ztf_name"]])

        else:
            n_spec.append(0)

    src_data["growth_n_spectra"] = n_spec

    growth_class_df = pd.read_csv(growth_class_path)

    g_class = []

    for _, row in tqdm(src_data.iterrows(), total=len(src_data)):
        mask = growth_class_df["name"] == row["ztf_name"]

        if np.sum(mask) > 0:
            classification = growth_class_df["comment"][mask].tolist()[0]
            g_class.append(classification)
        else:
            g_class.append(None)

    src_data["growth_class"] = g_class

    return src_data
