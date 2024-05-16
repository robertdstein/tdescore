"""
Module for augmenting data.
"""
import pandas as pd
from astropy.coordinates import SkyCoord

from tdescore.combine.parse_ps1 import parse_ps1
from tdescore.download import panstarrs_path
from tdescore.download.mast import download_panstarrs_data


def alert_to_pandas(alert) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a ZTF alert to a pandas dataframe

    :param alert: alert data
    :return: Pandas dataframe of detections
    """
    candidate = alert[0]["candidate"]
    prv_candid = alert[0]["prv_candidates"]
    combined = [candidate]
    combined.extend(prv_candid)

    df_detections_list = []
    df_ulims_list = []

    for cand in combined:
        _df = pd.DataFrame().from_dict(cand, orient="index").transpose()
        _df["mjd"] = _df["jd"] - 2400000.5
        if "magpsf" in cand.keys() and "isdiffpos" in cand.keys():
            df_detections_list.append(_df)

        else:
            df_ulims_list.append(_df)

    df_detections = pd.concat(df_detections_list)
    if len(df_ulims_list) > 0:
        df_ulims = pd.concat(df_ulims_list)
    else:
        df_ulims = None

    return df_detections, df_ulims


def augment_alerts(alert_data: dict):
    """
    Function to augment data for a table of sources.
    For...reasons, not all alerts have distpsnr1, so we need to set that manually

    :param alert_data: List of dictionaries of source data
    """

    df, _ = alert_to_pandas([alert_data])

    best = df.iloc[0].copy()
    best["ra"] = df["ra"].median()
    best["dec"] = df["dec"].median()
    best["ztf_name"] = alert_data["objectId"]

    best_df = pd.DataFrame(best).T

    if not panstarrs_path(best["ztf_name"]).exists():
        download_panstarrs_data(best_df)

    match = parse_ps1(alert_data["objectId"])

    match_c = SkyCoord(ra=match["ra_ps1"], dec=match["dec_ps1"], unit="deg")

    coord = SkyCoord(ra=df["ra"], dec=df["dec"], unit="deg")

    sep = coord.separation(match_c).to("arcsec").value

    df["distpsnr1"] = sep

    mask = df["jd"] == alert_data["candidate"]["jd"]
    assert mask.sum() == 1, f"Multiple detections at jd {alert_data['candidate']['jd']}"

    alert_data["candidate"]["distpsnr1"] = df[mask]["distpsnr1"].values[0]

    alert_data["candidate"]["ra_ps1"] = match["ra_ps1"]
    alert_data["candidate"]["dec_ps1"] = match["dec_ps1"]

    for prv_cand in alert_data["prv_candidates"]:
        if "magpsf" in prv_cand.keys():
            mask = df["jd"] == prv_cand["jd"]
            assert mask.sum() == 1, f"Multiple detections at jd {prv_cand['jd']}"
            prv_cand["distpsnr1"] = df[mask]["distpsnr1"].values[0]
            prv_cand["ra_ps1"] = match["ra_ps1"]
            prv_cand["dec_ps1"] = match["dec_ps1"]

    return alert_data
