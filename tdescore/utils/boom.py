"""
API queries with BOOM
"""

from blastwave import Source


def load_by_name(source_name: str, t_max_jd: float | None = None) -> list[dict]:
    """
    Load BOOM source data by source name

    :param source_name: source name
    :param t_max_jd: maximum JD to filter photometry
    :return: Raw alert data in ZTF style
    """
    source = Source.from_parquet(source_name)
    return [source.convert_to_ztfstyle()]
