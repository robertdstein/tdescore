"""
Module for downloading Fritz data
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import backoff
import pandas as pd
import requests
from tqdm import tqdm

from tdescore.paths import fritz_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)

# Fritz API URLs

API_BASEURL = "https://fritz.science"

fritz_token = os.getenv("FRITZ_TOKEN")


def fritz_api(
    method: str, endpoint_extension: str, data: dict = None
) -> requests.Response:
    """
    Function to query the Fritz API

    :param method: Method to use (get or post)
    :param endpoint_extension: Endpoint extension
    :param data: Data to send
    :return: Response
    """

    assert fritz_token is not None, "FRITZ_TOKEN bash variable is not set"

    headers = {"Authorization": f"token {fritz_token}"}

    endpoint = os.path.join(API_BASEURL, endpoint_extension)
    if method in ["post", "POST"]:
        response = requests.request(method, endpoint, json=data, headers=headers)
    elif method in ["get", "GET"]:
        response = requests.request(method, endpoint, params=data, headers=headers)
    else:
        raise ValueError("You have to use either 'get' or 'post'")
    return response


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=60,
)
def query_fritz_source(source_name: str) -> requests.Response:
    """
    Function to query the Fritz API for a source

    :param source_name: Name of source
    :return: Response
    """

    data = {
        "includeSpectrumExists": True,
    }
    return fritz_api(
        method="get", endpoint_extension=f"api/sources/{source_name}", data=data
    )


def fritz_path(source_name: str) -> Path:
    """
    Get path to fritz json cache

    :param source_name: Name of source
    :return: path
    """
    return fritz_cache_dir.joinpath(f"{source_name}.json")


def download_fritz_data(
    src_table: Optional[pd.DataFrame] = None,
):
    """
    Function to download Fritz crossmatch data for a table of sources

    :param src_table: Table of sources
    :return: None
    """
    logger.info("Downloading Fritz data")

    if fritz_token is None:
        logger.warning("FRITZ_TOKEN bash variable is not set, skipping download")

    else:
        if src_table is None:
            src_table = load_raw_sources()

        for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
            output_path = fritz_path(row["ztf_name"])

            if not output_path.exists():
                res_api = query_fritz_source(row["ztf_name"])

                if res_api.status_code == 200:
                    res = res_api.json()["data"]

                    with open(output_path, "w", encoding="utf8") as out_f:
                        out_f.write(json.dumps(res))
