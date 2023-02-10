"""
Module to analyse a lightcurve and extract metaparameters for further analysis
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from tdescore.alerts import get_lightcurve_vectors, load_source_clean, load_source_raw
from tdescore.classifications import all_source_list
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.extract import (
    extract_alert_parameters,
    extract_crossmatch_parameters,
    extract_lightcurve_parameters,
)
from tdescore.lightcurve.gaussian_process import fit_two_band_lightcurve
from tdescore.lightcurve.plot import plot_lightcurve_fit
from tdescore.paths import metadata_dir

logger = logging.getLogger(__name__)

TIME_KEY = "mjd"
Y_KEY = "magpsf"
YERR_KEY = "sigmapsf"


def get_metadata_path(source: str) -> Path:
    """
    Returns the unique metadata patf for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return metadata_dir.joinpath(f"{source}.json")


def has_enough_detections(lc_1: pd.DataFrame, lc_2: pd.DataFrame) -> bool:
    """
    Boolean check to see if dataset contains enough detections

    :param lc_1: primary color band
    :param lc_2: secondary data band
    :return: boolean true if enough data
    """

    n_tot = len(lc_1) + len(lc_2)

    if n_tot < 7:
        logger.debug(f"<7 datapoints ({n_tot}")
        return False

    if len(lc_2) < 3:
        logger.debug("No second band data for color")
        return False

    if len(lc_1) < 3:
        logger.debug(f"Too few primary datapoints ({len(lc_1)})")
        return False

    return True


def analyse_source_lightcurve(source: str, create_plot: bool = True):
    """
    Perform a full lightcurve analysis on a 2-band lightcurve.

    Fits the primary band with a gaussian process.
    Then fits a linear color evolution to map the second band to this model.
    Finally, refits the combined dataset.
    Uses the combined fit to extract lightcurve metadata, and saves this in json format

    :param source: ZTF source to analyse
    :param create_plot: boolean whether to save plot of lightcurve
    :return: None
    """

    clean_alert_data = load_source_clean(source)

    lc_g, lc_r, mag_offset = get_lightcurve_vectors(clean_alert_data)

    if not has_enough_detections(lc_1=lc_g, lc_2=lc_r):
        raise InsufficientDataError("Too few detections in data")

    gp_combined, lc_combined, popt = fit_two_band_lightcurve(lc_1=lc_g, lc_2=lc_r)

    param_dict, txt = extract_lightcurve_parameters(
        gp_combined=gp_combined,
        lc_combined=lc_combined,
        popt=popt,
        mag_offset=mag_offset,
    )

    param_dict.update(extract_alert_parameters(load_source_raw(source)))

    param_dict.update(extract_crossmatch_parameters(source))

    output_path = get_metadata_path(source)
    with open(output_path, "w", encoding="utf8") as out_f:
        out_f.write(json.dumps(param_dict))

    if create_plot:
        plot_lightcurve_fit(
            source=source,
            gp_combined=gp_combined,
            lc_1=lc_g,
            lc_2=lc_r,
            mag_offset=mag_offset,
            popt=popt,
            txt=txt,
        )


def batch_analyse(sources: Optional[list[str]] = None, overwrite: bool = False):
    """
    Iteratively analyses a batch of sources

    :param sources: list of source names
    :param overwrite: boolean whether to overwrite existing files
    :return: None
    """

    if sources is None:
        sources = all_source_list

    logger.info(f"Analysing {len(sources)} sources")

    failures = []
    data_missing = []

    for source in tqdm(sources):
        logger.debug(f"Analysing {source}")
        if not np.logical_and(get_metadata_path(source).exists(), not overwrite):
            try:
                analyse_source_lightcurve(source, create_plot=True)
            except InsufficientDataError:
                data_missing.append(source)
            except ValueError:
                failures.append(source)

    print(f"Insufficient data for {len(data_missing)} sources")
    print(f"Failed for {len(failures)} sources")
