"""
Combine BOOM crossmatches from Gaia
"""

import numpy as np
import pandas as pd
from blastwave import Source


def parse_redshift_crossmatch(source: Source) -> dict:
    """
    Parse Gaia crossmatch data from a Source object.

    :param source: Source object containing Gaia crossmatch data
    :return: Source object with parsed Gaia crossmatch data
    """

    match_dict = {
        "zspec": np.nan,
        "zphot": np.nan,
        "zllim": np.nan,
        "zorigin": None,
        "z_dist_arcsec": np.nan,
    }

    if source.crossmatches is not None:
        if "NED" in source.crossmatches:
            if len(source.crossmatches["NED"]) > 0:

                match = (
                    pd.DataFrame(source.crossmatches["NED"])
                    .sort_values(by="distance_arcsec")
                    .reset_index(drop=True)
                    .iloc[0]
                )
                match = match.replace({None: np.nan})

                if match["z_tech"].lower() == "phot":
                    match_dict["zphot"] = match["z"]
                    match_dict["zorigin"] = "NED_photz"
                    match_dict["z_dist_arcsec"] = match["distance_arcsec"]
                elif match["z_tech"].lower() == "spec":
                    match_dict["zspec"] = match["z"]
                    match_dict["zphot"] = match["z"]
                    match_dict["zllim"] = match["z"]
                    match_dict["zorigin"] = "NED_specz"
                    match_dict["z_dist_arcsec"] = match["distance_arcsec"]
                else:
                    match_dict["zphot"] = match["z"]
                    match_dict["zorigin"] = f"NED_{match['z_tech'].lower()}"
                    match_dict["z_dist_arcsec"] = match["distance_arcsec"]

        if pd.isnull(match_dict["zspec"]) and "DESI_DR1" in source.crossmatches:
            if len(source.crossmatches["DESI_DR1"]) > 0:

                match = (
                    pd.DataFrame(source.crossmatches["DESI_DR1"])
                    .sort_values(by="distance_arcsec")
                    .reset_index(drop=True)
                    .iloc[0]
                )
                match = match.replace({None: np.nan})

                match_dict["zspec"] = match["z"]
                match_dict["zphot"] = match["z"]
                match_dict["zllim"] = match["z"] - 2.0 * match["zerr"]
                match_dict["zorigin"] = "DESI_DR1_specz"
                match_dict["z_dist_arcsec"] = match["distance_arcsec"]

        if pd.isnull(match_dict["zspec"]) and "LS_DR10_PHOTOZ" in source.crossmatches:
            if len(source.crossmatches["LS_DR10_PHOTOZ"]) > 0:
                match = (
                    pd.DataFrame(source.crossmatches["LS_DR10_PHOTOZ"])
                    .sort_values(by="distance_arcsec")
                    .reset_index(drop=True)
                    .iloc[0]
                )
                match = match.replace({None: np.nan})

                if match["z_phot"] > -0.0:
                    match_dict["zphot"] = match["z_phot"]
                    match_dict["zllim"] = match["z_phot"] - 2.0 * match["z_phot_err"]
                    match_dict["zorigin"] = "LS_DR10_photz"
                    match_dict["z_dist_arcsec"] = match["distance_arcsec"]

    return match_dict
