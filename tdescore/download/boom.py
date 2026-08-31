"""
Download from BOOM
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from blastwave import BOOM_POOL_MAXSIZE, ZTFClient
from blastwave.utils import get_source_path
from tqdm.contrib.concurrent import thread_map

from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)


def download_boom(
    src_table: Optional[pd.DataFrame] = None,
    overwrite: bool = True,
):
    logger.info("Downloading BOOM data")

    if src_table is None:
        src_table = load_raw_sources()

    client = ZTFClient()

    def download_boom_single(
        name: str,
    ):
        """
        Function to download BOOM data for a single source

        :param name: Name of source
        :return: None
        """
        path = get_source_path(name)

        if overwrite or not path.exists():
            s = client.get_source(name)
            s.to_parquet()

    names = src_table["name"].tolist()

    thread_map(download_boom_single, names, max_workers=BOOM_POOL_MAXSIZE)
