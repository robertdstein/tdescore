"""
Module to analyse a lightcurve and extract metaparameters for further analysis
"""
import contextlib
import logging
import multiprocessing
from asyncio import timeout
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from tdescore.classifications import all_source_list
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
    THERMAL_WINDOWS,
    analyse_source_thermal,
    get_thermal_lightcurve_path,
)
from tdescore.lightcurve.week import analyse_source_week_data, get_week_lightcurve_path
from tdescore.paths import lightcurve_dir

logger = logging.getLogger(__name__)


def batch_analyse_thermal(
    sources: list[str],
    overwrite: bool = False,
    base_output_dir: Path = lightcurve_dir,
    save_resampled: bool = False,
    thermal_windows: list[float] = None,
):
    """
    Batch analysis of thermal data

    :param sources:
    :param overwrite:
    :param base_output_dir:
    :param save_resampled:
    :param thermal_windows:
    :return:
    """

    if thermal_windows is None:
        thermal_windows = THERMAL_WINDOWS

    lc_thermal_dir = base_output_dir.parent / "gp_thermal"
    lc_thermal_dir.mkdir(exist_ok=True)

    # Use a simplified Gaussian Process model for source
    # (using all available data rather than cleaning it up first)
    for source in sources:
        for window in thermal_windows:
            lc_output_dir = lc_thermal_dir / str(window)
            lc_output_dir.mkdir(exist_ok=True)
            if not np.logical_and(
                get_thermal_lightcurve_path(source, window).exists(), not overwrite
            ):
                analyse_source_thermal(
                    source,
                    base_output_dir=lc_output_dir,
                    save_resampled=save_resampled,
                    window_days=window,
                )


def analyse_single(
    source: str,
    overwrite: bool = False,
    base_output_dir: Path = lightcurve_dir,
    include_text: bool = True,
    save_resampled: bool = False,
    thermal_windows: Optional[list[float]] = None,
):
    base_output_dir = Path(base_output_dir)

    logger.debug(f"Analysing {source}")

    # Use only early data for source
    if not np.logical_and(get_infant_lightcurve_path(source).exists(), not overwrite):
        analyse_source_early_data(source)

    # Use only first week data for source
    if not np.logical_and(get_week_lightcurve_path(source).exists(), not overwrite):
        analyse_source_week_data(source)

    # Use only first month data for source
    if not np.logical_and(get_month_lightcurve_path(source).exists(), not overwrite):
        analyse_source_month_data(
            source,
            base_output_dir=base_output_dir,
        )

    # Use full lightcurve data for source
    if not np.logical_and(get_lightcurve_metadata_path(source).exists(), not overwrite):
        analyse_source_lightcurve(
            source,
            create_plot=True,
            base_output_dir=base_output_dir,
            include_text=include_text,
        )

    # Analyse thermal data
    if thermal_windows is not None:
        if len(thermal_windows) > 0:
            batch_analyse_thermal(
                sources=[source],
                overwrite=overwrite,
                base_output_dir=base_output_dir,
                save_resampled=save_resampled,
                thermal_windows=thermal_windows,
            )


def process_source(x):
    with contextlib.redirect_stdout(None):
        analyse_single(**x)


def batch_analyse(
    sources: Optional[list[str]] = None,
    overwrite: bool = False,
    base_output_dir: Path = lightcurve_dir,
    include_text: bool = True,
    save_resampled: bool = False,
    thermal_windows: Optional[list[float]] = None,
    timeout_duration: Optional[int] = None,
):
    """
    Iteratively analyses a batch of sources

    :param sources: list of source names
    :param overwrite: boolean whether to overwrite existing files
    :param base_output_dir: output directory for plots
    :param include_text: boolean whether to include text in plots
    :param save_resampled: boolean whether to save resampled data
    :param thermal_windows: list of thermal windows to use
    :param timeout_duration: timeout duration for each source
    :return: None
    """

    if sources is None:
        sources = all_source_list[::-1]

    logger.info(f"Analysing {len(sources)} sources, using {n_cpu} CPUs")

    source_kwargs = [
        {
            "source": source,
            "overwrite": overwrite,
            "base_output_dir": str(base_output_dir),
            "include_text": include_text,
            "save_resampled": save_resampled,
            "thermal_windows": thermal_windows,
        }
        for source in sources
    ]

    completed = []
    failed = []

    with multiprocessing.Pool(processes=1) as pool:
        results = [
            pool.apply_async(process_source, args=(kwargs,)) for kwargs in source_kwargs
        ]

        with tqdm(total=len(source_kwargs)) as progress_bar:
            for i, result in enumerate(results):
                try:
                    result.get(timeout=timeout_duration)
                    completed.append(source_kwargs[i]["source"])
                except multiprocessing.TimeoutError:
                    logger.warning(f"Timeout for {source_kwargs[i]['source']}")
                    failed.append(source_kwargs[i]["source"])
                finally:
                    progress_bar.update(1)

        logger.info(f"Completed {len(completed)} sources")
        logger.info(f"Failed {len(failed)} sources due to timeout")
