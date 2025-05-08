"""
Module to run sncosmo on sources
"""
import json
import logging
from typing import Optional

import numpy as np
import sncosmo
from tqdm import tqdm

from tdescore.classifications import all_source_list
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.paths import sn_cosmo_plot_dir, sncosmo_dir
from tdescore.sncosmo.utils import convert_df_to_table
from tdescore.lightcurve.window import get_window_data, THERMAL_WINDOWS
from pathlib import Path

model = sncosmo.Model(source="salt2")  # pylint: disable=no-member

FIT_PARAMS = ["z", "t0", "x0", "x1", "c"]

logger = logging.getLogger(__name__)


def get_sncosmo_path(source: str, window_days: float | None) -> Path:
    """
    Returns the unique sncosmo path for a particular source

    :param source: Source name
    :param window_days: Number of days to consider
    :return: path of sncosmo json
    """
    if window_days is None:
        window_days = "full"

    output_dir = sncosmo_dir.joinpath(f"{window_days}")
    output_dir.mkdir(exist_ok=True)

    return output_dir / f"{source}.json"


def get_sncosmo_plot_path(source: str, window_days: float | None) -> Path:
    """
    Returns the unique sncosmo path for a particular source

    :param source: Source name
    :param window_days: Number of days to consider
    :return: path of sncosmo json
    """
    if window_days is None:
        window_days = "full"

    output_dir = sn_cosmo_plot_dir.joinpath(f"{window_days}")
    output_dir.mkdir(exist_ok=True)

    return output_dir / f"{source}.pdf"


def sncosmo_fit(source: str, window_days: float | None, create_plot: bool = True):
    """
    Load clean data for a source, and fit SNIa SALT-2 models to it using sncosmo

    :param source: Name of source
    :param window_days: Number of days to consider
    :param create_plot: boolean whether to plot and save figure
    :return: None
    """
    # raw_df = load_source_clean(source)
    raw_df, _, _ = get_window_data(source, window_days=window_days, include_fp=True)

    label = f"sncosmo_{window_days}"


    try:
        if len(raw_df) < 3:
            raise InsufficientDataError("Too few datapoints")

        data = convert_df_to_table(raw_df.copy())

        result, fitted_model = sncosmo.fit_lc(  # pylint: disable=no-member
            data,
            model,
            FIT_PARAMS,
            bounds={"z": (0.0, 0.3)},  # parameters of model to vary
        )

        res = {}
        for i, param in enumerate(FIT_PARAMS):
            res[f"{label}_{param}"] = result.parameters[i]

        res[f"{label}_chisq"] = result.chisq
        res[f"{label}_ndof"] = result.ndof
        res[f"{label}_success"] = result.success
        res[f"{label}_ncall"] = result.ncall
        try:
            res[f"{label}_chi2pdof"] = result.chisq / result.ndof
        except ZeroDivisionError:
            res[f"{label}_chi2pdof"] = np.nan

        res[f"{label}_chi2overn"] = result.chisq / len(raw_df)

        output_path = get_sncosmo_path(source, window_days=window_days)
        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(res))

        if create_plot:
            sncosmo.plot_lc(  # pylint: disable=no-member
                data,
                model=fitted_model,
                errors=result.errors,
                fname=get_sncosmo_plot_path(source, window_days=window_days),
            )

    except InsufficientDataError:
        logger.warning(f"Insufficient data for {source} to run sncosmo")


def batch_sncosmo(sources: Optional[list[str]] = None, overwrite: bool = False):
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

    for window in THERMAL_WINDOWS:

        logger.info(f"Applying sncosmo with a cut of {window} days")

        for source in tqdm(sources):
            logger.debug(f"Analysing {source}")
            if not np.logical_and(get_sncosmo_path(source, window_days=window).exists(), not overwrite):
                try:
                    sncosmo_fit(source, window_days=window, create_plot=True)
                except InsufficientDataError:
                    data_missing.append(source)
                except (
                    ValueError,
                    RuntimeError,
                    sncosmo.fitting.DataQualityError,
                    KeyError,
                ):
                    failures.append(source)

        logger.info(f"Failed for {len(failures)} sources")
