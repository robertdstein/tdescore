"""
Combine results for each crossmatch with BOOM
"""

import pandas as pd
from blastwave import Source
from tqdm import tqdm

from tdescore.combine.boom.catwise import parse_catwise_crossmatch
from tdescore.combine.boom.gaia import parse_gaia_crossmatch
from tdescore.combine.boom.milliquas import parse_milliquas_crossmatch
from tdescore.combine.boom.redshift import parse_redshift_crossmatch
from tdescore.combine.boom.vsx import parse_vsx_crossmatch

all_parse_fs = [
    parse_catwise_crossmatch,
    parse_gaia_crossmatch,
    parse_milliquas_crossmatch,
    parse_vsx_crossmatch,
    parse_redshift_crossmatch,
]


def parse_boom_single_source(name: str) -> dict:
    """
    Parse a single source from BOOM

    :param name: Name of source
    :return: Dictionary with parsed source data
    """
    s = Source.from_parquet(name)
    res = {"name": name}

    for key in [
        "ndethist",
        "ndetfilters",
        "jdstarthist",
        "jdendhist",
        "age",
        "peak_mag",
    ]:
        res[key] = getattr(s, key)

    for parse_f in all_parse_fs:
        res.update(parse_f(s))
    return res


def parse_all_sources_boom(raw_source_table: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all sources from BOOM

    :param raw_source_table: Dataframe with raw source data
    :return: Dataframe with combined source data
    """
    res = []
    for name in tqdm(raw_source_table["name"].tolist()):
        res.append(parse_boom_single_source(name))

    match_df = pd.DataFrame(res)

    join_df = pd.concat(
        [
            raw_source_table.reset_index(drop=True),
            match_df[
                [x for x in match_df.columns if x not in raw_source_table.columns]
            ],
        ],
        axis=1,
    )

    return join_df
