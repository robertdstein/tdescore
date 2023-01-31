"""
Module to analyse a lightcurve and extract metaparameters for further analysis
"""
import json
import logging

import pandas as pd

from tdescore.alerts import get_lightcurve_vectors, load_source_clean, load_source_raw
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.extract import (
    extract_alert_parameters,
    extract_lightcurve_parameters,
)
from tdescore.lightcurve.gaussian_process import fit_two_band_lightcurve
from tdescore.lightcurve.plot import plot_lightcurve_fit
from tdescore.paths import metadata_dir

logger = logging.getLogger(__name__)

TIME_KEY = "mjd"
Y_KEY = "magpsf"
YERR_KEY = "sigmapsf"


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

    if len(lc_2) < 1:
        logger.debug("No second band data for color")
        return False

    if len(lc_1) < 3:
        logger.debug(f"Too few primary datapoints ({len(lc_1)})")
        return False

    return True


def analyse_source(source: str, create_plot: bool = True):
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

    lc_r, lc_g, mag_offset = get_lightcurve_vectors(clean_alert_data)

    if not has_enough_detections(lc_1=lc_g, lc_2=lc_r):
        raise InsufficientDataError("Too few detections in data")

    gp_combined, lc_combined, popt = fit_two_band_lightcurve(lc_1=lc_g, lc_2=lc_g)

    param_dict, txt = extract_lightcurve_parameters(
        gp_combined=gp_combined,
        lc_combined=lc_combined,
        popt=popt,
    )

    param_dict.update(extract_alert_parameters(load_source_raw(source)))

    output_path = metadata_dir.joinpath(f"{source}.json")
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