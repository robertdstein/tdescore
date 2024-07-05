"""
Module to run sncosmo on sources
"""
import json
import logging
from typing import Optional

import numpy as np
import sncosmo
from tqdm import tqdm

from tdescore.alerts import load_source_clean
from tdescore.classifications import all_source_list
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.paths import sn_cosmo_plot_dir, sncosmo_dir
from tdescore.sncosmo.utils import convert_df_to_table

model = sncosmo.Model(source="salt2")  # pylint: disable=no-member

FIT_PARAMS = ["z", "t0", "x0", "x1", "c"]

logger = logging.getLogger(__name__)


def get_sncosmo_path(source: str):
    """
    Returns the unique sncosmo path for a particular source

    :param source: Source name
    :return: path of sncosmo json
    """
    return sncosmo_dir.joinpath(f"{source}.json")


def get_sncosmo_plot_path(source: str):
    """
    Returns the unique sncosmo plot path for a particular source

    :param source: Source name
    :return: path of sncosmo figure
    """
    return sn_cosmo_plot_dir.joinpath(f"{source}.pdf")


def sncosmo_fit(source: str, create_plot: bool = True):
    """
    Load clean data for a source, and fit SNIa SALT-2 models to it using sncosmo

    :param source: Name of source
    :param create_plot: boolean whether to plot and save figure
    :return: None
    """
    raw_df = load_source_clean(source)

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
            res["sncosmo_" + param] = result.parameters[i]

        res["sncosmo_chisq"] = result.chisq
        res["sncosmo_ndof"] = result.ndof
        res["sncosmo_success"] = result.success
        res["sncosmo_ncall"] = result.ncall
        try:
            res["sncosmo_chi2pdof"] = result.chisq / result.ndof
        except ZeroDivisionError:
            res["sncosmo_chi2pdof"] = np.nan

        res["sncosmo_chi2overn"] = result.chisq / len(raw_df)

        output_path = get_sncosmo_path(source)
        with open(output_path, "w", encoding="utf8") as out_f:
            out_f.write(json.dumps(res))

        if create_plot:
            sncosmo.plot_lc(  # pylint: disable=no-member
                data,
                model=fitted_model,
                errors=result.errors,
                fname=get_sncosmo_plot_path(source),
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

    for source in tqdm(sources):
        logger.debug(f"Analysing {source}")
        if not np.logical_and(get_sncosmo_path(source).exists(), not overwrite):
            try:
                sncosmo_fit(source, create_plot=True)
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
