"""
This module contains functions to interact with the Kowalski API.
"""

import os

from astropy.time import Time
from penquins import Kowalski

fp_mapping = {"mag": "magpsf", "magerr": "sigmapsf"}

kwargs = {
    "protocol": "https",
    "host": "kowalski.caltech.edu",
    "port": 443,
    "verbose": False,
    "timeout": 300.0,
}

kowalski_token = os.getenv("KOWALSKI_TOKEN")


def download_kowalski_alert_data(
    ztf_name: str,
    t_max_jd: float | None = None,
    kowalski: Kowalski | None = None,
):
    """
    Download alert data from Kowalski

    :param ztf_name: Name of source
    :param t_max_jd: Maximum JD to query
    :param kowalski: Kowalski object
    :return: Alert data
    """
    if kowalski is None:
        kowalski = Kowalski(token=kowalski_token, **kwargs)

    if t_max_jd is None:
        t_max_jd = Time.now().jd

    query_config = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {
                "objectId": {"$eq": ztf_name},
                "candidate.isdiffpos": {"$in": ["1", "t", "true", "True", "T"]},
                "candidate.jd": {"$lt": t_max_jd},
            },
            "projection": {
                "_id": 0,
                "cutoutScience": 0,
                "cutoutTemplate": 0,
                "cutoutDifference": 0,
                "coordinates": 0,
            },
        },
    }

    query_result = kowalski.query(query_config)

    if "data" in query_result:
        alerts = query_result["data"]
    else:
        alerts = query_result.get("default").get("data")

    jds = [x["candidate"]["jd"] for x in alerts]

    max_idx = jds.index(max(jds))
    latest_alert = alerts[max_idx]

    jds = [latest_alert["candidate"]["jd"]]

    prv_alerts = []

    for prv_cand in alerts:
        if (prv_cand["candidate"]["jd"] not in jds) & (
            "magpsf" in prv_cand["candidate"]
        ):
            jds.append(prv_cand["candidate"]["jd"])
            prv_alerts.append(prv_cand["candidate"])

    # Query for prv_candidates/forced photometry

    query_config = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts_aux",
            "filter": {
                "_id": {"$eq": ztf_name},
            },
            "projection": {"cross_matches": 0},
        },
    }

    query_result = kowalski.query(query_config)
    if "data" in query_result:
        out = query_result["data"]
    else:
        out = query_result.get("default").get("data")

    if len(out) > 0:
        for prv_cand in out[0]["prv_candidates"]:
            if (
                (prv_cand["jd"] not in jds)
                & (prv_cand["jd"] < t_max_jd)
                & ("magpsf" in prv_cand)
            ):
                jds.append(prv_cand["jd"])
                prv_alerts.append(prv_cand)

        # Add forced photometry

        fp_dets = []

        if "fp_hists" in out[0]:
            for fp_dict in out[0]["fp_hists"]:
                if (
                    (fp_dict["jd"] not in jds)
                    & (fp_dict["jd"] < t_max_jd)
                    & ("mag" in fp_dict)
                    & ("magerr" in fp_dict)
                ):
                    if fp_dict["snr"] > 3.0:
                        for old_key, new_key in fp_mapping.items():
                            fp_dict[new_key] = fp_dict.pop(old_key)
                        fp_dict["isdiffpos"] = "t"
                        fp_dict["fp_bool"] = True
                        fp_dets.append(fp_dict)

        jds += [x["jd"] for x in fp_dets]
        prv_alerts += fp_dets

    latest_alert["prv_candidates"] = prv_alerts

    if not len(jds) == len(set(jds)):
        raise ValueError(f"Duplicate JDs for {ztf_name}")
    return [latest_alert]
