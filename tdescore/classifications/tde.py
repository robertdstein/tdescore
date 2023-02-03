"""
Module containing lists of ZTF TDEs
"""
import numpy as np
import pandas as pd

from tdescore.paths import data_dir

# ZTF I TDEs from Erica
ztf1_tde_list = pd.read_table(
    data_dir.joinpath("tde_names.txt"),
    skiprows=1,
    names=["AT Name", "ZTF Name", "Other1", "Other2", "Other3", "GOT Name", "spectype"],
)
ztf_i_tdes = ztf1_tde_list["ZTF Name"].tolist()

# Current ZTF II spreadsheet

ztf_ii_tde_path = data_dir.joinpath("ZTF-II TDEs - Current TDEs.csv")
ztf_ii_sources = pd.read_csv(ztf_ii_tde_path)
mask = np.array([("(" in x) | ("---" in x) for x in ztf_ii_sources["SWG Name"]])
ztf_ii_sources = ztf_ii_sources[~mask]
ztf_ii_tdes = ztf_ii_sources["ZTF Name"].tolist()

# Yao 23 list

yao_23_path = data_dir.joinpath("yao_23.dat")
yao_23_tdes = pd.read_csv(yao_23_path, sep=" ")["ztfname"].tolist()

all_tdes = sorted(list(set(ztf_i_tdes + ztf_ii_tdes + yao_23_tdes)))


def is_tde(names: np.ndarray) -> np.ndarray:
    """
    Boolean whether a given source is a known TDE

    :param names: names of sources
    :return: boolean mask array
    """
    return np.array([x in all_tdes for x in names])
