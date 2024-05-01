"""
Module to download generic WISE data
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
from astroquery.irsa import Irsa
from tqdm import tqdm

from tdescore.download.gaia import NpEncoder
from tdescore.paths import wise_cache_dir
from tdescore.raw import load_raw_sources

logger = logging.getLogger(__name__)

all_columns = [("neowiser_p1bs_psd", "mpro"), ("allwise_p3as_mep", "mpro_ep")]


def wise_path(source_name: str) -> Path:
    """
    Get path to WISE json cache

    :param source_name: Name of source
    :return: path
    """
    return wise_cache_dir.joinpath(f"{source_name}.json")


def clean_photometry(data):
    """
    Clean WISE data

    :param data: Astropy table
    :return: Good data
    """
    mask = (data["nb"] < 2) & (data["na"] < 1) & (data["cc_flags"] == "0000")
    return data[mask]


def download_wise_data(
    src_table: Optional[pd.DataFrame] = None,
    search_radius: float = 2.0,
):
    """
    Function to download WISE crossmatch data for a table of sources

    :param src_table: Table of sources
    :param search_radius: Search radius (arcsec)
    :return: None
    """
    logger.info("Downloading WISE data")

    if src_table is None:
        src_table = load_raw_sources()

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = wise_path(row["ztf_name"])

        if output_path.exists():
            continue

        coord = SkyCoord(
            ra=row["ra"], dec=row["dec"], unit=(u.degree, u.degree), frame="icrs"
        )

        radius = u.Quantity(search_radius, u.arcsec)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            allwise = Irsa.query_region(  # pylint: disable=no-member
                coord, catalog="allwise_p3as_psd", radius=radius
            )

        res = {}

        try:
            res = dict(allwise[0])

            all_r = Table()

            for column, key in all_columns:
                irsa_r = Irsa.query_region(  # pylint: disable=no-member
                    coord, catalog=column, radius=radius
                )
                r_cut = clean_photometry(irsa_r)

                if len(r_cut) > 0:
                    for j in range(2):
                        r_cut[f"w{j + 1}"] = r_cut[f"w{j + 1}{key}"]
                        r_cut[f"w{j + 1}_sigma"] = r_cut[f"w{j + 1}sig{key}"]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", AstropyWarning)
                        all_r = vstack([all_r, r_cut])

            for j in range(2):
                key = f"w{j + 1}"
                err_key = f"w{j + 1}_sigma"

                if len(all_r) > 1:
                    diff = np.sum(
                        (((all_r[key] - np.mean(all_r[key])) / all_r[err_key]) ** 2.0)
                        / (len(all_r) - 1)
                    )

                else:
                    diff = np.nan

                res[f"w{j + 1}_chi2"] = diff

        except (KeyError, IndexError):
            logger.debug(f"No match for {row['ztf_name']}")

        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(res, cls=NpEncoder))
