"""
Module to download generic WISE data
"""

import json
import logging
from pathlib import Path
from typing import Optional

import backoff
import pandas as pd
import requests
from dl import queryClient as qc
from tqdm import tqdm

from tdescore.paths import catwise_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)

all_columns = [("neowiser_p1bs_psd", "mpro"), ("allwise_p3as_mep", "mpro_ep")]


def catwise_path(source_name: str) -> Path:
    """
    Get path to WISE json cache

    :param source_name: Name of source
    :return: path
    """
    return catwise_cache_dir.joinpath(f"{source_name}.json")


@backoff.on_exception(
    backoff.expo,
    (qc.queryClientError, requests.exceptions.ReadTimeout),
    max_tries=5,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: logger.warning(
        f"Retrying catWISE query: {details['tries']} tries, "
        f"{details['wait']} seconds wait"
    ),
)
def get_catwise(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 3.0,
) -> pd.Series:
    """
    Function to query the catWISE database for photometry data

    :param ra_deg: RA in degrees
    :param dec_deg: Dec in degrees
    :param radius_arcsec: Search radius in arcseconds

    :return: DataFrame with the first match
    """
    radius_deg = radius_arcsec / 3600.0
    res = qc.query(
        sql=f"SELECT * from catwise2020.main "
        f"where 't' = Q3C_RADIAL_QUERY(ra, dec, {ra_deg}, {dec_deg}, {radius_deg}) "
        f"LIMIT 1",
        fmt="pandas",
    )

    if len(res) == 0:
        return pd.Series()

    return res.iloc[0]


def download_catwise_data(
    src_table: Optional[pd.DataFrame] = None,
    search_radius: float = 2.0,
):
    """
    Function to download catWISE crossmatch data for a table of sources

    :param src_table: Table of sources
    :param search_radius: Search radius (arcsec)
    :return: None
    """
    logger.info("Downloading catWISE data")

    if src_table is None:
        src_table = load_raw_sources()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = catwise_path(row["ztf_name"])

        if output_path.exists():
            continue

        catalog_data = get_catwise(row["ra"], row["dec"], search_radius)

        if len(catalog_data) == 0:
            res = json.dumps({})
        else:
            res = catalog_data.to_json()

        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(res)
