"""
Module for parsing data from caches, and copying all of it
"""

import numpy as np
import pandas as pd

from tdescore.combine.parse_full import parse_full
from tdescore.lightcurve.gaussian_process import MINIMUM_NOISE_MAGNITUDE
from tdescore.lightcurve.thermal import THERMAL_WINDOWS, get_thermal_lightcurve_path


def parse_all_thermal(source_name: str) -> dict:
    """
    Iteratively loop over each cache for a source which needs to be parsed in full

    :param source_name: Name of source
    :return: dictionary of results
    """

    res = {}

    for window in THERMAL_WINDOWS:
        base = parse_full(
            source_name,
            output_f=lambda x: get_thermal_lightcurve_path(x, window_days=window),
        )

        w = window if window is not None else 365.  # Fixme: Check this

        for field in ["rise", "fade"]:
            old_key = f"thermal_{field}"
            try:
                val = base[old_key]
            except KeyError:
                val = np.nan
            padded_val = val if pd.notnull(val) else w*2.
            base[f"thermal_{field}_padded"] = padded_val

        label = f"thermal_{window}d"

        new = {}

        for key in base:
            if label not in key:
                new[key.replace("thermal", label)] = base[key]
            else:
                new[key] = base[key]

        res.update(new)

        try:
            res[f"{label}_high_noise"] = (
                res[f"{label}_noise"] > MINIMUM_NOISE_MAGNITUDE + 0.01
            )
        except KeyError:
            res[f"{label}_high_noise"] = np.nan

    return res
