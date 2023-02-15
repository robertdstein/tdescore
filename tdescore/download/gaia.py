"""
Module to download generic Gaia data
"""
import json
import logging
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from tqdm import tqdm

from tdescore.paths import gaia_cache_dir
from tdescore.raw import load_raw_sources

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Ensure Data Release 3

logger = logging.getLogger(__name__)


# Thanks StackOverflow!
class NpEncoder(json.JSONEncoder):
    """
    Encoder which handles the weird astropy table types
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def download_gaia_data(
    src_table: Optional[pd.DataFrame] = None,
    search_radius: float = 1.5,
):
    """
    Function to download Gaia DR3 crossmatch data for a table of sources

    :param src_table: Table of sources
    :param search_radius: Search radius (arcsec)
    :return: None
    """
    logger.info("Downloading Gaia data")

    if src_table is None:
        src_table = load_raw_sources()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):

        output_path = gaia_cache_dir.joinpath(f"{row['ztf_name']}.json")

        if not output_path.exists():

            Gaia.ROW_LIMIT = 1  # Ensure the default row limit.

            coord = SkyCoord(
                ra=row["ra"], dec=row["dec"], unit=(u.degree, u.degree), frame="icrs"
            )

            radius = u.Quantity(search_radius, u.arcsec)

            job = Gaia.cone_search(coord, radius)
            res_table = job.get_results()
            if len(res_table) == 0:
                res = {}
            else:
                res = dict(res_table[0])

            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(res, cls=NpEncoder))
