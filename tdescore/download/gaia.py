"""
Module to download generic Gaia data
"""

import json
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from tdescore.paths import gaia_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)


class GaiaError(Exception):
    """
    Generic Gaia error
    """


def check_server(gaia_instance) -> bool:
    """
    Check if the Gaia server is responsive and functional

    :param gaia_instance: Gaia instance
    :return: boolean whether server is responsive
    """
    conn_handler = gaia_instance._TapPlus__getconnhandler()

    default_server_ok = conn_handler.get_response_status() == 200

    if default_server_ok:
        sub_context = gaia_instance.GAIA_MESSAGES
        response = conn_handler.execute_tapget(sub_context, verbose=False)

        for line in response:
            try:
                line.decode("utf-8").split("=", 1)[1]
            except IndexError:
                default_server_ok = False
                logger.warning("Gaia server response invalid")
                break

    else:
        logger.warning("Could not connect to Gaia server")

    return default_server_ok


def get_gaia():
    """
    Get a Gaia instance, handling server issues
    """

    with redirect_stdout(open(os.devnull, "w")):
        from astroquery.gaia import Gaia as gaia

    default_server_ok = check_server(gaia)

    # if not default_server_ok:
    #     from astroquery.gaia import GaiaClass
    #     from astroquery.utils.tap.core import TapPlus
    #     gaia = TapPlus(url="http://gaia.ari.uni-heidelberg.de/tap/")
    #
    #     default_server_ok = check_server(gaia)
    #     print(default_server_ok)
    #
    #     if not default_server_ok:
    #
    #         url = "https://gaia.aip.de/tap"
    #         gaia = TapPlus(url=url)
    #         # gaia.cone_search()
    #         print(check_server(gaia))
    # #         raise
    #
    #         # raise ConnectionError("Both Gaia servers are down")
    #     # Gaia = GaiaClass(
    #     #     gaia_tap_server='http://gaia.ari.uni-heidelberg.de/tap/',
    #     #     gaia_data_server='http://gaia.ari.uni-heidelberg.de/tap/',
    #     # )

    if not default_server_ok:
        raise GaiaError("Gaia server is down")

    gaia.ROW_LIMIT = 1  # Ensure the default row limit.
    gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    return gaia


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


def gaia_path(source_name: str) -> Path:
    """
    Get path to Gaia json cache

    :param source_name: Name of source
    :return: path
    """
    return gaia_cache_dir.joinpath(f"{source_name}.json")


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

    try:

        gaia = get_gaia()

        if src_table is None:
            src_table = load_raw_sources()

        for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
            output_path = gaia_path(row["ztf_name"])

            if not output_path.exists():

                coord = SkyCoord(
                    ra=row["ra"],
                    dec=row["dec"],
                    unit=(u.degree, u.degree),
                    frame="icrs",
                )

                radius = u.Quantity(search_radius, u.arcsec)

                job = gaia.cone_search(coordinate=coord, radius=radius)
                res_table = job.get_results()
                if len(res_table) == 0:
                    res = {}
                else:
                    res = dict(res_table[0])

                with open(output_path, "w", encoding="utf8") as out_f:
                    out_f.write(json.dumps(res, cls=NpEncoder))

    except GaiaError as exc:
        logger.error(f"Could not download Gaia data: {exc}")
        raise exc
