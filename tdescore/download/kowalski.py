"""
Interface to Kowalski for downloading data
"""
import json
import logging
import os
from pathlib import Path

import pandas as pd
from penquins import Kowalski
from tqdm import tqdm

from tdescore.download.gaia import NpEncoder
from tdescore.paths import ps1strm_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)

PROTOCOL, HOST, PORT = "https", "kowalski.caltech.edu", 443


def get_kowalski() -> Kowalski | None:
    """
    Get a Kowalski object, using credentials stored in the environment

    :return: Kowalski object
    """

    token_kowalski = os.environ.get("KOWALSKI_TOKEN")

    if token_kowalski is not None:
        kowalski_instance = Kowalski(
            token=token_kowalski, protocol=PROTOCOL, host=HOST, port=str(PORT)
        )

    else:
        logger.info("No Kowalski token found, skipping Kowalski queries.")
        return None

    if not kowalski_instance.ping():
        err = "Error connecting to Kowalski. Are your credentials right?"
        raise ValueError(err)

    return kowalski_instance


def near_query_kowalski(
    kowalski: Kowalski,
    ra_deg: float,
    dec_deg: float,
    catalog_name: str,
    search_radius_arcsec: float = 1.5,
    limit_match: int = 1,
) -> dict:
    """
    Performs a Kowalski query around coords

    :param kowalski: Kowalski instance
    :param ra_deg: RA in degrees
    :param dec_deg: Dec in degrees
    :param catalog_name: Name of catalog to query
    :param search_radius_arcsec: Search radius in arcsec
    :param limit_match: Limit of matches
    :return: crossmatch dict
    """
    query = {
        "query_type": "near",
        "query": {
            "max_distance": search_radius_arcsec,
            "distance_units": "arcsec",
            "radec": {"query_coords": [ra_deg, dec_deg]},
            "catalogs": {f"{catalog_name}": {}},
        },
        "kwargs": {
            "max_time_ms": 10000,
            "limit": limit_match,
        },
    }
    response = kowalski.query(query=query)
    try:
        data = response.get("default").get("data")[catalog_name]["query_coords"][0]
    except (KeyError, IndexError):
        data = {}
    return data


def ps1strm_path(source_name: str) -> Path:
    """
    Returns path of panstarrs cache for source

    :param source_name: name of source
    :return: path to cache
    """
    return ps1strm_cache_dir.joinpath(f"{source_name}.json")


def download_ps1strm_data(src_table: pd.DataFrame | None = None):
    """
    Function to download ps1strm source data for each source in a table

    :param src_table: Table of sources
    :return: None
    """
    if src_table is None:
        src_table = load_raw_sources()

    kowalski = get_kowalski()
    if kowalski is None:
        logger.warning("No Kowalski instance found, skipping download")
        return

    logger.info("Downloading ps1strm data")
    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = ps1strm_path(row["ztf_name"])

        if not output_path.exists():
            res = near_query_kowalski(
                kowalski,
                ra_deg=row["ra"],
                dec_deg=row["dec"],
                catalog_name="PS1_STRM",
                limit_match=1,
            )

            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(res, cls=NpEncoder))
