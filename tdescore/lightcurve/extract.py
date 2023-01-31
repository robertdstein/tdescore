"""
Module for extracting features for ML classification
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from tdescore.alerts import get_positive_detection_mask
from tdescore.data import get_crossmatch
from tdescore.lightcurve.color import linear_color

ALERT_COPY_KEYS = ["sgscore1"]


# pylint: disable=R0914
def extract_lightcurve_parameters(
    gp_combined: GaussianProcessRegressor,
    lc_combined: pd.DataFrame,
    popt: np.ndarray,
) -> tuple[dict, str]:
    """
    Extract lightcurve metaparameters from a lightcurve fit

    :param gp_combined: gaussian process fit
    :param lc_combined: combined lightcurve data
    :param popt: color fit parameters
    :return: dictionary of metaparameters, and human-readable text summary
    """

    param_dict = {}

    score = gp_combined.score(
        lc_combined["time"].to_numpy(dtype=float).reshape(-1, 1),
        lc_combined["magpsf"].to_numpy(dtype=float).reshape(-1, 1),
    )
    txt = f"Score: {score:.2f}, "
    param_dict["score"] = score

    length_scale = gp_combined.kernel_.get_params()["k1__k2__length_scale"]
    txt += f"Length Scale: {length_scale:.2f} days, "
    param_dict["length_scale"] = length_scale

    y_scale = gp_combined.kernel_.get_params()["k1__k1__constant_value"]
    txt += f"Y Scale: {y_scale:.2f} \n"
    param_dict["y_scale"] = y_scale

    n_infs = []

    delta = 0.5
    t_50s = []

    t_range = np.linspace(min(lc_combined["time"]), max(lc_combined["time"]), 1000)
    y_pred_raw = gp_combined.predict(t_range.reshape(-1, 1))

    y_peak = max(y_pred_raw)

    t_peak_g = t_range[y_pred_raw == y_peak]

    for i, mask in enumerate([t_range < t_peak_g, t_range > t_peak_g]):

        n_inf = 0
        t_50 = np.nan

        if np.sum(mask) > 1:
            smooth = y_pred_raw[mask]

            # compute second derivative
            smooth_d2 = np.gradient(smooth)

            # find switching points
            infls = np.where(np.diff(np.sign(smooth_d2)))[0]

            n_inf = len(infls)

            y_mask = smooth > (y_peak - delta)

            if np.sum(~y_mask) > 0:
                t_50 = (
                    (t_range[mask][~y_mask][i - 1] - t_peak_g[0])
                    * 2
                    * (np.sign(i) - 0.5)
                )

        txt += (
            f"{['pre-', 'post-'][i]}peak lightcurve has {n_inf} inflection points,"
            f" {['rise', 'fade'][i]} = {t_50:.1f} d \n"
        )

        t_50s.append(t_50)

        n_infs.append(n_inf)

    param_dict["rise"] = t_50s[0]
    param_dict["pre_inflection"] = n_infs[0]
    param_dict["fade"] = t_50s[1]
    param_dict["post_inflection"] = n_infs[1]

    txt += (
        f"Color at peak: {float(linear_color(t_peak_g, *popt)):.2f} mag, "
        f"color grad: {1000. * popt[0]:.2f} milli-mag/day \n"
    )

    param_dict["peak_color"] = float(linear_color(t_peak_g, *popt))
    param_dict["color_grad"] = 1000.0 * popt[0]

    n_det = len(lc_combined)
    param_dict["n_det"] = n_det

    density = n_det / max(lc_combined["time"])
    cadence = 1.0 / density
    param_dict["det_cadence"] = cadence

    txt += f"n_det: {n_det}, density = {cadence:.2f} days between detections  "

    return param_dict, txt


def extract_alert_parameters(raw_alert_data: pd.DataFrame) -> dict:
    """
    Extract various metaparameters from raw alert data

    :param raw_alert_data: raw alert data
    :return: relevant metaparameters
    """
    param_dict = {}

    positive_mask = get_positive_detection_mask(raw_alert_data)
    positive_fraction = np.mean(positive_mask.astype(float))

    param_dict["positive_fraction"] = positive_fraction

    # clean_data = clean_source(raw_alert_data)

    for key in ALERT_COPY_KEYS:
        param_dict[key] = float(raw_alert_data.iloc[0][key])

    return param_dict


def extract_crossmatch_parameters(source_name: str) -> dict:
    """
    Extract various metaparameters from crossmatches

    :param source_name: name of source
    :return: relevant metaparameters
    """
    crossmatch = get_crossmatch(source_name).to_dict("records")
    assert len(crossmatch) == 1
    return crossmatch[0]
