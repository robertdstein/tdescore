"""
Module for parsing data from caches, and copying all of it
"""

from tdescore.combine.parse_partial import parse_subset_of_cache
from tdescore.lightcurve.window import THERMAL_WINDOWS
from tdescore.sncosmo.run_sncosmo import get_sncosmo_path

BASE_KEYS = [
    "chisq",
    "chi2pdof",
    "x1",
    "c",
    # Ones that are not used by the classifier
    "chi2overn",
    "ndof",
    "z",
    "x0",
]

def get_sncosmo_keys(window: float) -> list[str]:
    """
    Get the keys for a particular window

    :param window: Window of days
    :return: list of keys
    """

    return [f"sncosmo_{window}_{key}" for key in BASE_KEYS]

def parse_sncosmo(source_name: str) -> dict:
    """
    Iteratively loop over each cache for a source which needs to be parsed in full

    :param source_name: Name of source
    :return: dictionary of results
    """

    res = {}

    for window in THERMAL_WINDOWS:

        keys = get_sncosmo_keys(window)

        base = parse_subset_of_cache(
            source_name,
            output_f=lambda x: get_sncosmo_path(x, window_days=window),
            copy_keys=keys,
        )

        res.update(base)

    return res
