"""
Module for performing milliquas crossmatching
"""
import pandas as pd
from tqdm import tqdm

from tdescore.classifications.crossmatch import crossmatch_with_catalog
from tdescore.paths import data_dir

miliquas_path = data_dir.joinpath("milliquas.txt")

if not miliquas_path.exists():
    raise FileNotFoundError(
        f"No file found at {miliquas_path}, "
        f"try downloading at https://quasars.org/milliquas.zip"
    )

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
    "cite",
    "zcite",
    "rxpct",
    "qpct",
    "xname",
    "rname",
    "lobe1",
    "lobe2",
]

colspecs = [
    (0, 10),
    (12, 22),
    (25, 49),
    (51, 54),
    (56, 60),
    (62, 66),
    (68, 70),
    (72, 73),
    (74, 75),
    (76, 81),
    (83, 88),
    (90, 95),
    (97, 99),
    (101, 103),
    (105, 126),
    (128, 149),
    (151, 172),
    (174, 195),
]


def crossmatch_to_milliquas(src_data) -> pd.DataFrame:
    """
    Crossmatch to milliquas catalog

    :param src_data: array of all source ra values
    :return: updated_src_data
    """

    mq_data = pd.read_fwf(miliquas_path, names=names, colspecs=colspecs)

    match_bool = []
    scores = []
    m_class = []

    for _, row in tqdm(src_data.iterrows(), total=len(src_data)):
        match = crossmatch_with_catalog(
            catalog=mq_data, ra_deg=row["ra"], dec_deg=row["dec"]
        )

        match_bool.append(int(match is not None))
        if match is not None:
            scores.append(match["qpct"])
            m_class.append(match["type"])
        else:
            scores.append(-1.0)
            m_class.append(None)

    src_data["has_milliquas"] = match_bool
    src_data["milliquas_score"] = scores
    src_data["milliquas_class"] = m_class

    return src_data
