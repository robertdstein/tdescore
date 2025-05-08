"""
Module containing lists of ZTF TDEs
"""
import logging

import numpy as np
import pandas as pd

from tdescore.paths import data_dir

logger = logging.getLogger(__name__)

ztf_i_path = data_dir.joinpath("tde_names.txt")

try:
    # ZTF I TDEs
    ztf1_tde_list = pd.read_table(
        ztf_i_path,
        skiprows=1,
        names=[
            "AT Name",
            "ZTF Name",
            "Other1",
            "Other2",
            "Other3",
            "GOT Name",
            "spectype",
        ],
    )
    ztf_i_tdes = ztf1_tde_list["ZTF Name"].tolist()

except FileNotFoundError:
    logger.warning(f"No ZTF I TDE list found at {ztf_i_path}. Setting to empty list.")
    ztf_i_tdes = []

ztf_ii_tde_path = data_dir.joinpath("ZTF-II TDEs - Current TDEs.csv")
# Altname
if not ztf_ii_tde_path.exists():
    ztf_ii_tde_path = data_dir.joinpath("ZTFII.csv")

try:
    # Current ZTF II spreadsheet
    ztf_ii_sources = pd.read_csv(ztf_ii_tde_path)
    mask = np.array([("(" in x) | ("---" in x) for x in ztf_ii_sources["SWG Name"]])
    ztf_ii_sources = ztf_ii_sources[~mask]
    ztf_ii_tdes = ztf_ii_sources["ZTF Name"].tolist()
except FileNotFoundError:
    logger.warning(
        f"No ZTF II TDE list found at {ztf_ii_tde_path}. Setting to empty list."
    )
    ztf_ii_tdes = []

# Yao 23 list
yao_23_path = data_dir.joinpath("yao_23.dat")
try:
    yao_23_tdes = pd.read_csv(yao_23_path, sep=" ")["ztfname"].tolist()
except FileNotFoundError:
    logger.warning(f"No Yao 23 list found at {yao_23_path}. Setting to empty list.")
    yao_23_tdes = []

non_tdes = ["ZTF18aasvknh", "ZTF18acpdvos"]  # Bad lightcurve - TDE in reference image

all_tdes = sorted(
    [x for x in set(ztf_i_tdes + ztf_ii_tdes + yao_23_tdes) if x not in non_tdes]
)


def is_tde(names: np.ndarray) -> np.ndarray[bool]:
    """
    Boolean whether a given source is a known TDE

    :param names: names of sources
    :return: boolean mask array
    """
    return np.array([x in all_tdes for x in names])
