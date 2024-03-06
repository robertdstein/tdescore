"""
Module for downloading TNS data
"""
import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import backoff
import pandas as pd
import requests
from tqdm import tqdm

from tdescore.paths import tns_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)

API_BASEURL = "https://ampel.zeuthen.desy.de/api/catalogmatch"


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=600,
)
def ampel_api_catalog(
    catalog: str,
    catalog_type: str,
    ra_deg: float,
    dec_deg: float,
    search_radius_arcsec: float = 3.0,
    search_type: str = "all",
) -> requests.Response:
    """
    Method for querying catalogs via the Ampel API
    'catalog' must be the name of a supported catalog, e.g.
    SDSS_spec, PS1, NEDz_extcats...
    For a full list of catalogs, confer
    https://ampel.zeuthen.desy.de/api/catalogmatch/catalogs

    :param catalog: Name of catalog
    :param catalog_type: Type of catalog
    :param ra_deg: RA in degrees
    :param dec_deg: Dec in degrees
    :param search_radius_arcsec: Search radius in arcseconds
    :param search_type: Search type

    """
    if catalog_type not in ["extcats", "catsHTM"]:
        raise ValueError(
            f"Expected parameter catalog_type in ['extcats', 'catsHTM'], got {catalog_type}"
        )
    if search_type not in ["all", "nearest"]:
        raise ValueError(
            f"Expected parameter catalog_type in ['all', 'nearest'], got {search_type}"
        )

    queryurl_catalogmatch = API_BASEURL + "/cone_search/" + search_type

    # First, we create a json body to post
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    query = {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "catalogs": [
            {"name": catalog, "rs_arcsec": search_radius_arcsec, "use": catalog_type}
        ],
    }

    logger.debug(queryurl_catalogmatch)
    logger.debug(query)

    response = requests.post(url=queryurl_catalogmatch, json=query, headers=headers)

    return response


def ampel_tns_query(
    ra_deg: float, dec_deg: float, search_radius_arcsec: float = 3.0
) -> requests.Response:
    """
    Wrapper for ampel_api_catalog to query TNS

    :param ra_deg: RA in degrees
    :param dec_deg: Dec in degrees
    :param search_radius_arcsec: Search radius in arcseconds
    :return: Nearest TNS match
    """
    return ampel_api_catalog(
        catalog="TNS",
        catalog_type="extcats",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=search_radius_arcsec,
        search_type="nearest",
    )


def tns_path(source_name: str) -> Path:
    """
    Get path to tns json cache

    :param source_name: Name of source
    :return: path
    """
    return tns_cache_dir.joinpath(f"{source_name}.json")


def download_tns_data(
    src_table: Optional[pd.DataFrame] = None,
):
    """
    Function to download TNS crossmatch data for a table of sources

    :param src_table: Table of sources
    :return: None
    """
    logger.info("Downloading TNS data via Ampel")

    if src_table is None:
        src_table = load_raw_sources()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = tns_path(row["ztf_name"])

        if not output_path.exists():
            res_api = ampel_tns_query(row["ra"], row["dec"])

            if res_api.status_code == 200:
                res = res_api.json()[0]
                if res is None:
                    all_res = {}
                else:
                    all_res = res["body"]
                    all_res["dist_arcsec"] = res["dist_arcsec"]

                with open(output_path, "w", encoding="utf8") as out_f:
                    out_f.write(json.dumps(all_res))
