"""
Module to analyse a lightcurve and extract metaparameters for further analysis
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from tdescore.classifications import all_source_list
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.full import (
    analyse_source_lightcurve,
    get_lightcurve_metadata_path,
)
from tdescore.lightcurve.infant import (
    analyse_source_early_data,
    get_infant_lightcurve_path,
)
from tdescore.lightcurve.month import (
    analyse_source_month_data,
    get_month_lightcurve_path,
)
from tdescore.lightcurve.thermal import (
    analyse_source_thermal,
    get_thermal_lightcurve_path,
)
from tdescore.lightcurve.week import analyse_source_week_data, get_week_lightcurve_path
from tdescore.paths import lightcurve_dir

logger = logging.getLogger(__name__)


def batch_analyse(
    sources: Optional[list[str]] = None,
    overwrite: bool = False,
    base_output_dir: Path = lightcurve_dir,
    include_text: bool = True,
    save_resampled: bool = False,
):
    """
    Iteratively analyses a batch of sources

    :param sources: list of source names
    :param overwrite: boolean whether to overwrite existing files
    :param base_output_dir: output directory for plots
    :param include_text: boolean whether to include text in plots
    :param save_resampled: boolean whether to save resampled data
    :return: None
    """

    if sources is None:
        sources = all_source_list[::-1]

    logger.info(f"Analysing {len(sources)} sources")

    failures = []
    data_missing = []
    no_alert_data = []

    lc_thermal_dir = base_output_dir.parent / "gp_thermal"
    lc_thermal_dir.mkdir(exist_ok=True)

    for source in tqdm(sources):
        logger.debug(f"Analysing {source}")
        try:
            # Use a simplified Gaussian Process model for source
            # (using all available data rather than cleaning it up first)
            if not np.logical_and(
                get_thermal_lightcurve_path(source).exists(), not overwrite
            ):
                analyse_source_thermal(
                    source,
                    base_output_dir=lc_thermal_dir,
                    save_resampled=save_resampled,
                )

            # Use only early data for source
            if not np.logical_and(
                get_infant_lightcurve_path(source).exists(), not overwrite
            ):
                analyse_source_early_data(source)

            # Use only first week data for source

            if not np.logical_and(
                get_week_lightcurve_path(source).exists(), not overwrite
            ):
                analyse_source_week_data(source)

            # Use only first month data for source
            if not np.logical_and(
                get_month_lightcurve_path(source).exists(), not overwrite
            ):
                analyse_source_month_data(
                    source,
                    base_output_dir=base_output_dir,
                )

            # Use full lightcurve data for source
            if not np.logical_and(
                get_lightcurve_metadata_path(source).exists(), not overwrite
            ):
                analyse_source_lightcurve(
                    source,
                    create_plot=True,
                    base_output_dir=base_output_dir,
                    include_text=include_text,
                )

        except InsufficientDataError:
            data_missing.append(source)
        except (ValueError, KeyError, RuntimeError, IndexError):
            failures.append(source)

    logger.info(f"Insufficient data for {len(data_missing)} sources")
    logger.info(f"No alert data for {len(no_alert_data)} sources")
    logger.info(f"Failed for {len(failures)} sources")
