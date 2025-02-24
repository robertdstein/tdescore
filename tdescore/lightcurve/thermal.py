"""
Module for analysing full lightcurve data with simple model
"""
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy import units as u
from astropy.modeling.models import BlackBody
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor

from tdescore.classifications.crossmatch import get_classification
from tdescore.lightcurve.errors import InsufficientDataError
from tdescore.lightcurve.extinction import get_extinction_correction
from tdescore.lightcurve.full import extract_lightcurve_parameters
from tdescore.lightcurve.gaussian_process import get_gp_model
from tdescore.lightcurve.infant import analyse_window_data, offset_from_average_position
from tdescore.lightcurve.plot import FIG_HEIGHT, FIG_WIDTH
from tdescore.lightcurve.utils import get_covariance_ellipse
from tdescore.paths import (
    lightcurve_dir,
    lightcurve_resampled_dir,
    lightcurve_thermal_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_FILL_VALUE = np.nan

THERMAL_WINDOWS = [
    # 14.0, # FIXME: Uncomment this line
    30.0,
    60.0,
    90.0,
    180.0,
]

wavelengths = {
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
}

extra_wavelengths = {
    "UVW2": 2079.0,
    "U": 3465.0,
    "g": 4770.0,
    "J": 12350.0,
}

colors = {
    "g": "g",
    "r": "r",
    "i": "orange",
    "UVW2": "purple",
    "UVM2": "blue",
    "UVW1": "cyan",
    "U": "blue",
    "z": "brown",
    "Y": "black",
    "J": "brown",
}


def get_thermal_lightcurve_path(source: str, window_days: float) -> Path:
    """
    Returns the unique metadata path for a particular source

    :param source: Source name
    :param window_days: number of days to consider
    :return: path of metadata json
    """

    output_dir = lightcurve_thermal_dir.joinpath(f"{window_days}")
    output_dir.mkdir(exist_ok=True)

    return output_dir / f"{source}.json"


MIN_TEMPERATURE_K = 3.0e3


def get_temperature(
    data,
    log_temperature_initial: float,
    temperature_linear: float = 0.0,
    temperature_quadratic: float = 0.0,
    temperature_cubic: float = 0.0,
):
    """
    Get the temperature for the black body

    :param data:
    :param log_temperature_initial: Log10 of initial temperature
    :param temperature_linear: Linear temperature change
    :param temperature_quadratic: Quadratic temperature change
    :param temperature_cubic: Cubic temperature change
    :return: Temperature as a function of time
    """
    temperature_initial = 10.0**log_temperature_initial
    temperature = temperature_initial + (
        temperature_linear * data.T[0]
        + temperature_quadratic * data.T[0] ** 2.0
        + temperature_cubic * data.T[0] ** 3.0
    )
    temperature[temperature < MIN_TEMPERATURE_K] = MIN_TEMPERATURE_K
    return temperature


def black_body(data, *temp_parameters):
    """
    Linear rise function

    :param data: 2d array (time, wavelength)
    :param temp_parameters: temperature parameters
    :return: value
    """
    temperature = get_temperature(data, *temp_parameters)

    bb_model = BlackBody(temperature=temperature * u.K)

    flux = bb_model(data.T[1] * u.AA) * (4 * np.pi * u.sr)

    mag = flux.to(u.ABmag).value

    flux_g = bb_model(4770.0 * u.AA) * (4 * np.pi * u.sr)
    mag_g = flux_g.to(u.ABmag).value

    return mag_g - mag


def fit_thermal(
    lc_df: pd.DataFrame,
    gp_1: GaussianProcessRegressor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to take a Gaussian Process model of a bolometric lightcurve, and fit
    a linear color evolution which maps this to a second lightcurve band.

    :param lc_df: band lightcurve data
    :param gp_1: gaussian lightcurve trained on one band
    :return: thermal model
    """

    t_data = lc_df["time"].to_numpy().reshape(-1, 1)

    pred_g = gp_1.predict(t_data, return_std=False)

    def thermal_model(data, *temp_parameters):
        pred_r = pred_g.flatten() + black_body(data, *temp_parameters)
        return pred_r

    max_index = 1

    temp_bounds = (100.0, 10.0, 1.0)

    bounds = (
        (np.log10(MIN_TEMPERATURE_K), *[-x for x in temp_bounds[:max_index]]),
        (6.0, *temp_bounds[:max_index]),
    )

    best_fit, pcov = curve_fit(
        thermal_model,
        xdata=lc_df[["time", "wavelength"]].to_numpy(),
        ydata=lc_df["magpsf"],
        sigma=lc_df["sigmapsf"].to_numpy(dtype=float),
        p0=[4.0, 0.0, 0.0, 0.0][: max_index + 1],
        bounds=bounds,
    )

    return best_fit, pcov


def plot_thermal_fit(
    source: str,
    gp_combined: GaussianProcessRegressor,
    lc_df: pd.DataFrame,
    mag_offset: float,
    popt: np.ndarray,
    pcov: np.ndarray,
    base_output_dir: Path = lightcurve_dir,
):
    """
    Plot a lightcurve fit

    :param source: Source name
    :param gp_combined: Gaussian process model
    :param lc_df: lightcurve data
    :param mag_offset: offset from lightcurve transformation
    :param popt: optimal color parameters
    :param pcov: covariance matrix
    :param base_output_dir: output directory
    :return: None
    """
    classification = get_classification(source)

    out_dir = base_output_dir.joinpath(f"{str(classification).replace(' ', '_')}")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir.joinpath(f"{source}.png")

    title = f"{source}"
    if classification is not None:
        title += f" ({classification})"

    t_array = np.linspace(
        lc_df["time"].min(),
        lc_df["time"].max(),
        1000,
    )

    n_ax_x = 3
    n_ax_y = 3

    fig = plt.figure(figsize=(FIG_WIDTH * 3.0, FIG_HEIGHT * n_ax_y / 2.0))
    plt.suptitle(title)

    # Plot the lightcurve data + model in separate bands

    for i, band in enumerate(["g", "r", "i"]):
        ax = plt.subplot(n_ax_x, n_ax_y, n_ax_y * i + 1)
        mask = lc_df["filter"] == band

        wavelength = np.ones_like(t_array) * wavelengths[band]

        y_pred_raw, sigma = gp_combined.predict(t_array.reshape(-1, 1), return_std=True)

        y_pred_raw = y_pred_raw + black_body(np.array([t_array, wavelength]).T, *popt)

        y_pred = mag_offset - y_pred_raw

        ax.fill(
            np.concatenate([t_array, t_array[::-1]]),
            np.concatenate([y_pred - 1.0 * sigma, (y_pred + 1.0 * sigma)[::-1]]),
            alpha=0.3,
            fc=colors[band],
            ec="None",
        )
        ax.plot(t_array, y_pred, linestyle=":", color=colors[band])
        ax.scatter(
            lc_df[mask]["time"], mag_offset - lc_df[mask]["magpsf"], c=colors[band]
        )

        if i < len(wavelengths) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        ax.invert_yaxis()
        ax.set_xlabel("Time since peak [days]")
        ax.set_ylabel(rf"$m_{band}$")

    plt.setp(ax.get_xticklabels(), visible=True)

    # Plot the multi-band model in one panel

    ax = plt.subplot(2, 3, 2)

    for band, wavelength_aa in extra_wavelengths.items():
        wavelength = np.ones_like(t_array) * wavelength_aa

        y_pred_raw, sigma = gp_combined.predict(t_array.reshape(-1, 1), return_std=True)

        y_pred_raw = y_pred_raw + black_body(np.array([t_array, wavelength]).T, *popt)

        y_pred = mag_offset - y_pred_raw

        n_sigmas = np.linspace(0, 1.0, 20)

        for n_sigma in n_sigmas:
            ax.fill(
                np.concatenate([t_array, t_array[::-1]]),
                np.concatenate(
                    [y_pred - n_sigma * sigma, (y_pred + n_sigma * sigma)[::-1]]
                ),
                alpha=1.0 / len(n_sigmas),
                fc=colors[band],
                ec="None",
            )

        plt.plot(t_array, y_pred, color=colors[band], label=band)

    ax.legend(ncol=2)

    plt.xlabel("Time since peak [days]")

    ax.set_ylabel(r"$m$")

    plt.setp(ax.get_xticklabels(), visible=False)

    ax.invert_yaxis()

    # Plot the temperature model

    ax = plt.subplot(2, 3, 2 + n_ax_x)

    temps = get_temperature(np.array([t_array]).T, *popt)

    y_pred_raw, sigma = gp_combined.predict(t_array.reshape(-1, 1), return_std=True)

    ax.plot(t_array, temps, linestyle=":", color="black")

    idx_max = np.argmax(y_pred_raw)
    temp_at_peak = temps[idx_max]
    temp_range = np.max(temps) - np.min(temps)

    sign = np.sign(temp_at_peak - np.median(temps))

    ax.annotate(
        f"{temp_at_peak:.0f} K",
        xy=(t_array[idx_max], temps[idx_max]),
        xytext=(t_array[idx_max], temps[idx_max] - sign * 0.3 * temp_range),
        arrowprops={"facecolor": "black", "shrink": 0.05},
    )
    plt.xlabel("Time since peak [days]")
    plt.ylabel("Temperature [K]")

    sns.despine()
    plt.subplots_adjust(hspace=0.0)

    # Plot the covariance matrix

    ax = plt.subplot(2, 6, 5)

    log_temp, cooling = get_covariance_ellipse(popt, pcov)

    ax.scatter(10.0 ** popt[0], popt[1], linestyle=":", color="black", marker="*")
    ax.plot(10.0**log_temp, cooling, linestyle=":", color="black")

    plt.ylabel("Temperature Change [K/day]")
    plt.xlabel(r"Temperature At Peak [K]")

    plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)


def resample_and_export_lightcurve(
    source: str,
    df: pd.DataFrame,
    gp_combined: GaussianProcessRegressor,
    popt: np.ndarray,
    step=1.0,
):
    """
    Generate and save a resampled lightcurve data

    :param source: Source name
    :param df: lightcurve data
    :param gp_combined: Gaussian process model
    :param popt: Best fit parameters
    :param step: time step for resampling
    :return: None
    """

    times = df["time"].to_list() + [180.0]

    med_offet = []
    avg_offset = []

    for i in range(len(df) + 1):
        df_cut = df[: i + 1]
        med_offet.append(np.nanmedian(df_cut["distpsnr1"]))
        avg_offset.append(offset_from_average_position(df_cut))

    med_offet += med_offet[-1]
    avg_offset += avg_offset[-1]

    t_min = df["time"].min()
    t_max = df["time"].max()

    t_array = np.arange(t_min, t_min + 180.0, step=step)

    med_interp = np.interp(t_array, times, med_offet)

    avg_interp = np.interp(t_array, times, avg_offset)

    y_pred_raw = gp_combined.predict(t_array.reshape(-1, 1), return_std=False)

    mask = t_array > t_max

    y_pred_raw[mask] = np.nan

    new = {}

    for band, wavelength_aa in wavelengths.items():
        wavelength = np.ones_like(t_array) * wavelength_aa
        y_pred = y_pred_raw + black_body(np.array([t_array, wavelength]).T, *popt)
        new[band] = y_pred

    new["time"] = t_array - min(t_array)

    new["med_offset"] = med_interp
    new["avg_offset"] = avg_interp

    output_path = lightcurve_resampled_dir.joinpath(f"{source}.npz")

    np.savez(output_path, **new)


def analyse_source_thermal(
    source: str,
    base_output_dir: Optional[Path] = None,
    save_resampled: bool = False,
    window_days: float = 180.0,
):
    """
    Perform a lightcurve analysis on first detections

    :param source: ZTF source to analyse
    :param base_output_dir: output directory for plots (default None)
    :param save_resampled: boolean whether to save resampled data
    :param window_days: number of days to consider (default 180)
    :return: None
    """
    label = f"thermal_{window_days}d"

    try:
        df, _, new_values = analyse_window_data(
            source=source, window_days=window_days, label=label, include_fp=True
        )

        df["filter"] = df["fid"].map({1: "g", 2: "r", 3: "i"})
        df["wavelength"] = df["filter"].map(wavelengths)

        if (len(df) > 1) & (len(df["filter"].unique()) > 1):
            ra = df["ra"].mean()
            dec = df["dec"].mean()

            for wavelength in df["wavelength"].unique():
                mask = df["wavelength"] == wavelength
                if mask.sum() > 0:
                    ext = get_extinction_correction(
                        ra_deg=ra, dec_deg=dec, wavelengths=[wavelength]
                    )
                    df.loc[mask, ["magpsf"]] -= ext

            df["time"] = df["mjd"] - min(df["mjd"])

            df["magpsf"] = df["magpsf"].astype(float)

            mag_offset = max(df["magpsf"])

            df["magpsf"] = -df["magpsf"] + mag_offset

            # First pass - fit monochromatic Gaussian Process model

            initial_lc_fit = get_gp_model(
                df["time"].to_numpy(dtype=float),
                df["magpsf"].to_numpy(dtype=float),
            )

            tarray = np.linspace(
                df["time"].min(),
                df["time"].max(),
                1000,
            )

            y_pred_raw, _ = initial_lc_fit.predict(
                tarray.reshape(-1, 1), return_std=True
            )

            t_max = tarray[np.argmax(y_pred_raw)]

            # Shift time to peak

            df["time"] -= t_max

            # Refit Gaussian Process model

            initial_lc_fit = get_gp_model(
                df["time"].to_numpy(dtype=float),
                df["magpsf"].to_numpy(dtype=float),
            )

            # Fit thermal model on top of Gaussian Process model

            popt, _ = fit_thermal(lc_df=df, gp_1=initial_lc_fit)

            # Estimate g-band magnitude for each point

            df["mag_pseudo_g"] = df["magpsf"] - black_body(
                df[["time", "wavelength"]].to_numpy(), *popt
            )

            # Second pass - fit combined Gaussian Process model

            df["time"] = df["mjd"] - min(df["mjd"])

            gp_combined = get_gp_model(
                df["time"].to_numpy(dtype=float),
                df["mag_pseudo_g"].to_numpy(dtype=float),
            )

            y_pred_raw, _ = gp_combined.predict(tarray.reshape(-1, 1), return_std=True)
            t_max = tarray[np.argmax(y_pred_raw)]

            df["time"] -= t_max

            gp_combined = get_gp_model(
                df["time"].to_numpy(dtype=float),
                df["mag_pseudo_g"].to_numpy(dtype=float),
            )

            popt, pcov = fit_thermal(lc_df=df, gp_1=gp_combined)

            if base_output_dir is not None:
                plot_thermal_fit(
                    source=source,
                    gp_combined=gp_combined,
                    lc_df=df,
                    mag_offset=mag_offset,
                    popt=popt,
                    pcov=pcov,
                    base_output_dir=base_output_dir,
                )

            x_pos, y_pos = get_covariance_ellipse(popt, pcov)

            new_values["thermal_log_temp_peak"] = popt[0]
            new_values["thermal_log_temp_sigma"] = np.sqrt(pcov[0, 0])

            new_values["thermal_cooling"] = popt[1]
            new_values["thermal_cooling_sigma"] = np.sqrt(pcov[1, 1])
            new_values["thermal_cross_term"] = pcov[0, 1]

            new_values["thermal_log_temp_ll"] = min(x_pos)
            new_values["thermal_log_temp_ul"] = max(x_pos)

            new_values["thermal_cooling_ll"] = min(y_pos)
            new_values["thermal_cooling_ul"] = max(y_pos)

            res, _ = extract_lightcurve_parameters(
                gp_combined=gp_combined,
                lc_combined=df,
                popt=popt,
                mag_offset=mag_offset,
            )

            for key, value in res.items():
                if "color" not in key:
                    new_values[f"thermal_{key}"] = value

            output_path = get_thermal_lightcurve_path(source, window_days=window_days)
            with open(output_path, "w", encoding="utf8") as out_f:
                out_f.write(json.dumps(new_values))

            if save_resampled:
                resample_and_export_lightcurve(
                    source=source,
                    df=df,
                    gp_combined=gp_combined,
                    popt=popt,
                )

        else:
            raise InsufficientDataError(
                f"Too few detections in data for {source} to run full analysis"
            )

    except InsufficientDataError:
        logger.warning(f"Insufficient data for {source} and window {window_days}")
