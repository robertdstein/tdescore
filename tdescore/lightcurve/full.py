"""
Module for full lightcurve analysis.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from tdescore.alerts import get_lightcurve_vectors, load_source_clean
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.extinction import get_extinction_correction
from tdescore.lightcurve.extract import extract_lightcurve_parameters
from tdescore.lightcurve.gaussian_process import fit_two_band_lightcurve
from tdescore.lightcurve.plot import plot_lightcurve_fit
from tdescore.paths import lightcurve_dir, lightcurve_metadata_dir

logger = logging.getLogger(__name__)


def get_lightcurve_metadata_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_metadata_dir.joinpath(f"{source}.json")


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


def analyse_source_lightcurve(
    source: str,
    create_plot: bool = True,
    base_output_dir: Path = lightcurve_dir,
    include_text: bool = True,
):
    """
    Perform a full lightcurve analysis on a 2-band lightcurve.

    Fits the primary band with a gaussian process.
    Then fits a linear color evolution to map the second band to this model.
    Finally, refits the combined dataset.
    Uses the combined fit to extract lightcurve metadata, and saves this in json format

    :param source: ZTF source to analyse
    :param create_plot: boolean whether to save plot of lightcurve
    :param base_output_dir: output directory for plots
    :param include_text: boolean whether to include text in plots
    :return: None
    """

    clean_alert_data = load_source_clean(source)

    ra = clean_alert_data["ra"].mean()
    dec = clean_alert_data["dec"].mean()

    [ext_g, ext_r] = get_extinction_correction(ra_deg=ra, dec_deg=dec)

    lc_g, lc_r, mag_offset = get_lightcurve_vectors(clean_alert_data, ext_g, ext_r)

    if not has_enough_detections(lc_1=lc_g, lc_2=lc_r):
        raise InsufficientDataError("Too few detections in data")

    gp_combined, lc_combined, popt = fit_two_band_lightcurve(lc_1=lc_g, lc_2=lc_r)

    param_dict, txt = extract_lightcurve_parameters(
        gp_combined=gp_combined,
        lc_combined=lc_combined,
        popt=popt,
        mag_offset=mag_offset,
    )

    output_path = get_lightcurve_metadata_path(source)
    with open(output_path, "w", encoding="utf8") as out_f:
        out_f.write(json.dumps(param_dict))

    if create_plot:
        txt = txt if include_text else None
        plot_lightcurve_fit(
            source=source,
            gp_combined=gp_combined,
            lc_1=lc_g,
            lc_2=lc_r,
            mag_offset=mag_offset,
            popt=popt,
            txt=txt,
            base_output_dir=base_output_dir,
        )
