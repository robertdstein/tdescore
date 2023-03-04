"""
Util functions for converting things to sncosmo-friendly formats
"""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table

_zp = (1 * u.mJy).to(u.ABmag)


def get_flux(mag_array: np.ndarray) -> np.ndarray:
    """
    Function to get flux from an AB mag array

    :param mag_array: Array of AB mags
    :return: flux in mJy
    """
    return (mag_array * u.ABmag).to("mJy")


def convert_df_to_table(input_df: pd.DataFrame) -> Table:
    """
    Convert a pandas dataframe to an sncosmo-esque astropy table

    :param input_df: Input dataframe
    :return: Astropy Table
    """
    input_df["time"] = input_df["mjd"]
    bands = np.empty(len(input_df), dtype=object)

    for i, filter_n in enumerate(["g", "r", "i"]):
        mask = input_df["fid"] == i + 1

        bands[mask] = f"ztf{filter_n}"

    input_df["band"] = bands
    input_df["zpsys"] = "ab"
    #     df["zp"] = df["magzpsci"]
    input_df["zp"] = _zp.value

    flux = get_flux(input_df["magpsf"].to_numpy())

    err = get_flux((input_df["magpsf"] - input_df["sigmapsf"]).to_numpy()) - flux

    input_df["flux"] = flux.value
    input_df["fluxerr"] = err.value

    tab = Table.from_pandas(
        input_df[["time", "band", "flux", "fluxerr", "zp", "zpsys"]]
    )

    return tab
