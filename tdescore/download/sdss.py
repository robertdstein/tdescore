"""
Module for downloading SDSS data
"""
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
from tqdm import tqdm

from tdescore.download.gaia import NpEncoder
from tdescore.paths import sdss_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)


def sdss_path(source_name: str) -> Path:
    """
    Get path to SDSS json cache

    :param source_name: Name of source
    :return: path
    """
    return sdss_cache_dir.joinpath(f"{source_name}.json")


def download_sdss_data(
    src_table: Optional[pd.DataFrame] = None,
    search_radius: float = 2.0,
):
    """
    Function to download SDSS crossmatch data for a table of sources

    :param src_table: Table of sources
    :param search_radius: Search radius (arcsec)
    :return: None
    """
    logger.info("Downloading SDSS data")

    if src_table is None:
        src_table = load_raw_sources()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = sdss_path(row["ztf_name"])

        if not output_path.exists():
            coord = SkyCoord(
                ra=row["ra"], dec=row["dec"], unit=(u.degree, u.degree), frame="icrs"
            )

            # pylint: disable=no-member
            catalog_data = SDSS.query_region(
                coord,
                radius=search_radius * u.arcsec,
                spectro=True,
                specobj_fields=["ra", "dec", "class", "subclass", "z"],
            )

            if catalog_data is None:
                res = {}
            else:
                c_xy = SkyCoord(
                    ra=catalog_data["ra"],
                    dec=catalog_data["dec"],
                    unit=(u.degree, u.degree),
                    frame="icrs",
                )
                dists = coord.separation(c_xy).to(u.arcsec)
                catalog_data["dist_arcsec"] = dists
                res = dict(catalog_data.group_by("dist_arcsec")[0])

            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(res, cls=NpEncoder))
