"""
Module for collating metadata json files generated from lightcurve analysis
"""
import pandas as pd

from tdescore.paths import combined_metadata_path, metadata_dir


def parse_metadata():
    """
    Iteratively parse all metadata files, and combine them into a single dataframe.
    Save this dataframe.
    """
    paths = [x for x in metadata_dir.iterdir() if x.suffix in ".json"]

    combined_records = pd.DataFrame()

    for path in paths:
        combined_records = pd.concat(
            [combined_records, pd.read_json(path, orient="record", typ="series")],
            ignore_index=True,
            axis=1,
        )

    combined_records = combined_records.transpose()
    combined_records.to_json(combined_metadata_path)
