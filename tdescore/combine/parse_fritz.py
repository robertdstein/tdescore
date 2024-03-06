"""
Module for parsing cached data, and copying a subset of it
"""
import json

import pandas as pd

from tdescore.download import fritz_path

FRITZ_COPY_KEYS = ["redshift", "tns_name", "spectrum_exists"]


def parse_fritz(source_name: str) -> dict:
    """
    Function to extract a subset of all parameters from a source data cache

    :param source_name: Name of source
    :return: pandas dataframe
    """

    cache_path = fritz_path(source_name)

    res = {}

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf8") as cache_f:
            cat_res = json.load(cache_f)

        for key in FRITZ_COPY_KEYS:
            res[f"fritz_{key}"] = cat_res[key]

        classes = cat_res["classifications"]

        if len(classes) > 0:
            fritz_class_df = pd.DataFrame(classes)

            useful_class_mask = (
                ~(fritz_class_df["ml"].astype(bool))
                & (fritz_class_df["probability"] > 0.5)
                & (fritz_class_df["origin"] != "SCoPe")
            )

            fritz_class_df = fritz_class_df[useful_class_mask]
            fritz_class_df.sort_values(
                ["probability", "created_at"], ascending=False, inplace=True
            )

            class_list = fritz_class_df["classification"].tolist()

            class_set = sorted(set(class_list), key=class_list.index)

            label = ",".join(class_set)

        else:
            label = None

        res["fritz_current_class"] = label

    return res
