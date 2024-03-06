"""
Module for performing BTS crossmatching
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from tdescore.paths import data_dir

bts_path = data_dir.joinpath("bts_sources.csv")


def crossmatch_to_bts(src_data) -> pd.DataFrame:
    """
    Crossmatch to BTS catalog

    :param src_data: array of all source ra values
    :return: updated_src_data
    """

    if not bts_path.exists():
        raise FileNotFoundError(f"No file found at {bts_path}")

    bts_class_df = pd.read_csv(bts_path)

    bts_class = []

    for _, row in tqdm(src_data.iterrows(), total=len(src_data)):
        mask = bts_class_df["ZTFID"] == row["ztf_name"]

        if np.sum(mask) > 0:
            classification = bts_class_df["type"][mask].tolist()[0]
            bts_class.append(classification)
        else:
            bts_class.append(None)

    src_data["bts_class"] = bts_class

    return src_data
