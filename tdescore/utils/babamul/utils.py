"""
Util functions for Babamul data processing.
"""

import polars as pl

from tdescore.paths import babamul_cache
from tdescore.utils.babamul.models import Source


def load_dataframe():
    """
    Load Babamul source data as a dataframe

    :return: Babamul Source dataframe
    """
    df = pl.read_parquet(babamul_cache)
    return df


def get_source_by_name(source_name: str | int) -> Source:
    """
    Load Babamul source data by source name

    :param source_name: source name
    :return: Babamul Source object
    """
    df = load_dataframe()
    row = df.filter(pl.col("objectid") == str(source_name))
    if len(row) == 0:
        raise ValueError(f"Source {source_name} not found in Babamul data.")
    return Source(**row.to_dicts()[0])


def load_by_name(source_name: str, t_max_jd: float | None = None) -> list[dict]:
    """
    Load Babamul source data by source name

    :param source_name: source name
    :param t_max_jd: maximum JD to filter photometry
    :return: Raw alert data in ZTF style
    """
    source = get_source_by_name(source_name)
    return [source.convert_to_ztfstyle()]
