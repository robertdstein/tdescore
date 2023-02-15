"""
Module for downloading MAST catalog data
"""
import json
import logging
from pathlib import Path
from typing import Optional

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from tqdm import tqdm

from tdescore.download.gaia import NpEncoder
from tdescore.paths import panstarrs_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)


def download_mast_data(
    src_table: pd.DataFrame, cat: str, output_dir: Path, search_radius: float = 1.5
):
    """
    Function to download MAST crossmatch data for each source in a dataframe,
    and save each as a json file

    :param src_table: table of sources
    :param cat: MAST catalog name
    :param output_dir: output directory for json
    :param search_radius: search radius in arcsec
    :return: None
    """
    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):

        output_path = output_dir.joinpath(f"{row['ztf_name']}.json")

        if not output_path.exists():

            coord = SkyCoord(
                ra=row["ra"], dec=row["dec"], unit=(u.degree, u.degree), frame="icrs"
            )

            # pylint: disable=no-member
            catalog_data = Catalogs.query_region(
                coord, radius=search_radius / 3600.0, catalog=cat
            )

            if len(catalog_data) == 0:
                res = {}
            else:
                res = dict(catalog_data.group_by("distance")[0])

            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(res, cls=NpEncoder))


def download_panstarrs_data(src_table: Optional[pd.DataFrame] = None):
    """
    Function to download panstarrs source data for each source in a table

    :param src_table: Table of sources
    :return: None
    """
    if src_table is None:
        src_table = load_raw_sources()
    logger.info("Downloading Panstarrs data")
    download_mast_data(
        src_table=src_table,
        cat="Panstarrs",
        output_dir=panstarrs_cache_dir,
        search_radius=1.5,
    )
