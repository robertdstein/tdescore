"""
Module for collating metadata json files generated from lightcurve analysis
"""
import logging

import pandas as pd
from tqdm import tqdm

from tdescore.classifications.bts import crossmatch_to_bts
from tdescore.classifications.growth_marshal import crossmatch_to_growth
from tdescore.classifications.milliquas import crossmatch_to_milliquas
from tdescore.combine.parse_fritz import parse_fritz
from tdescore.combine.parse_full import parse_all_full
from tdescore.combine.parse_gaia import parse_gaia
from tdescore.combine.parse_partial import parse_all_partial
from tdescore.combine.parse_ps1 import parse_ps1
from tdescore.combine.parse_ps1strm import parse_ps1strm
from tdescore.combine.parse_sdss import parse_sdss
from tdescore.combine.parse_thermal import parse_all_thermal
from tdescore.combine.parse_tns import parse_tns
from tdescore.combine.parse_wise import parse_wise
from tdescore.combine.parse_catwise import parse_catwise
from tdescore.paths import combined_metadata_path

logger = logging.getLogger(__name__)

all_path_fs = [
    parse_all_thermal,
    parse_tns,
    parse_fritz,
    parse_all_full,
    parse_all_partial,
    parse_gaia,
    parse_ps1,
    parse_sdss,
    parse_wise,
    parse_ps1strm,
    parse_catwise,
]


def combine_single_source(source_name: str) -> pd.DataFrame:
    """
    Iteratively collect each set of data for a given source

    :param source_name: source to collate
    :return: dataframe
    """

    res = {"ztf_name": source_name}

    for parse_f in all_path_fs:
        res.update(parse_f(source_name))

    return pd.DataFrame.from_dict(res, orient="index")


def combine_all_sources(
    raw_source_table: pd.DataFrame, save: bool = True
) -> pd.DataFrame:
    """
    Takes a raw source table, loops over it,
    and collates additional data for each source

    :param raw_source_table: table of sources
    :param save: whether to save the metadata
    :return: updated source table
    """

    all_series = []

    for source in tqdm(raw_source_table["ztf_name"].tolist()):
        all_series.append(combine_single_source(source))

    combined_records = pd.concat(
        all_series,
        axis=1,
        ignore_index=True,
    ).transpose()

    full_dataset = raw_source_table.join(
        combined_records.set_index("ztf_name"), on="ztf_name", validate="1:1"
    )

    full_dataset = crossmatch_to_milliquas(full_dataset)
    try:
        full_dataset = crossmatch_to_growth(full_dataset)
    except FileNotFoundError:
        logger.warning("Growth Marshal data not found")

    try:
        full_dataset = crossmatch_to_bts(full_dataset)
    except FileNotFoundError:
        logger.warning("BTS data not found")

    if save:
        with open(combined_metadata_path, "w", encoding="utf8") as output_f:
            full_dataset.to_json(output_f)

    return full_dataset


def load_metadata() -> pd.DataFrame:
    """
    Function to load the aggregated metadata

    :return: metadata dataframe
    """

    return pd.read_json(combined_metadata_path)
