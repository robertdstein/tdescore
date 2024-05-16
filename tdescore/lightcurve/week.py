"""
Module for analysing early lightcurve data
"""
import json
from pathlib import Path

from tdescore.lightcurve.infant import analyse_window_data
from tdescore.paths import lightcurve_week_dir

DEFAULT_FILL_VALUE = 0.0


def get_week_lightcurve_path(source: str) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :return: path of metadata json
    """
    return lightcurve_week_dir.joinpath(f"{source}.json")


def analyse_source_week_data(source: str):
    """
    Perform a lightcurve analysis on first detections

    :param source: ZTF source to analyse
    :return: None
    """

    window_days = 7.0
    label = "week"

    early_alert_data, _, new_values = analyse_window_data(
        source=source, window_days=window_days, label=label
    )

    g_early = early_alert_data[early_alert_data["fid"] == 1]
    r_early = early_alert_data[early_alert_data["fid"] == 2]

    if len(g_early) > 0:
        new_values["week_g_rise"] = g_early["magpsf"].max() - g_early["magpsf"].min()

    else:
        new_values["week_g_rise"] = DEFAULT_FILL_VALUE

    if len(r_early) > 0:
        new_values["week_r_rise"] = r_early["magpsf"].max() - r_early["magpsf"].min()
    else:
        new_values["week_r_rise"] = DEFAULT_FILL_VALUE

    if len(g_early) > 0 and len(r_early) > 0:
        new_values["week_median_color"] = (
            g_early["magpsf"].median() - r_early["magpsf"].median()
        )
    else:
        new_values["week_median_color"] = DEFAULT_FILL_VALUE

    output_path = get_week_lightcurve_path(source)
    with open(output_path, "w", encoding="utf8") as out_f:
        out_f.write(json.dumps(new_values))
