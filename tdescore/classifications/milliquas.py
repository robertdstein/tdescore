"""
Module for performing milliquas crossmatching
"""
import pandas as pd
from tqdm import tqdm

from tdescore.classifications.crossmatch import crossmatch_with_catalog
from tdescore.paths import data_dir

miliquas_path = data_dir.joinpath("milliquas_v8.txt")

if not miliquas_path.exists():
    raise FileNotFoundError(
        f"No file found at {miliquas_path}, "
        f"try downloading at https://quasars.org/milliquas.zip"
    )

# Milliquas V8 columns
names = [
    "ra",
    "dec",
    "name",
    "type",
    "r_mag",
    "b_mag",
    "comment",
    "r",
    "b",
    "redshift",
    # "cite",
    # "zcite",
    # "rxpct",
    # "qpct",
    # "xname",
    # "rname",
    # "lobe1",
    # "lobe2",
]

colspecs = [
    (0, 11),
    (12, 22),
    (24, 49),
    (50, 53),
    (55, 59),
    (61, 65),
    (67, 69),
    (71, 72),
    (73, 74),
    (75, 80),
    # (82, 87),
    # (89, 94),
    # (96, 98),
    # (100, 102),
    # (104, 125),
    # (127, 148),
    # (150, 171),
    # (173, 194),
]


def crossmatch_to_milliquas(src_data) -> pd.DataFrame:
    """
    Crossmatch to milliquas catalog

    :param src_data: array of all source ra values
    :return: updated_src_data
    """

    mq_data = pd.read_fwf(miliquas_path, names=names, colspecs=colspecs, skiprows=6)[:-1]
    mq_data["dec"] = mq_data["dec"].astype(float)
    mq_data["ra"] = mq_data["ra"].astype(float)

    match_bool = []
    scores = []
    m_class = []

    for _, row in tqdm(src_data.iterrows(), total=len(src_data)):
        match = crossmatch_with_catalog(
            catalog=mq_data, ra_deg=float(row["ra"]), dec_deg=float(row["dec"])
        )

        match_bool.append(int(match is not None))
        if match is not None:
            scores.append(1.0)  # No more Qpct in V8
            m_class.append(match["type"])
        else:
            scores.append(-1.0)
            m_class.append(None)

    src_data["has_milliquas"] = match_bool
    src_data["milliquas_score"] = scores
    src_data["milliquas_class"] = m_class

    return src_data
