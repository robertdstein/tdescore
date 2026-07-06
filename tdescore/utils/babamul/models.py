import json
import logging

import numpy as np
import pandas as pd
from astropy import units as u
from babamul.api import get_object
from babamul.exceptions import APINotFoundError
from babamul.models import LsstAlert, ZtfAlert
from pydantic import AliasChoices, BaseModel, ValidationError, computed_field

from tdescore.paths import babamul_cache_dir
from tdescore.utils.kowalski import download_kowalski_alert_data

logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)

FID_MAPPING = {
    "g": "1",
    "r": "2",
    "i": "3",
    "u": "-1",
    "z": "-2",
    "y": "-3",
}


class Observation(BaseModel):
    jd: float
    magpsf: float
    sigmapsf: float
    diffmaglim: float
    ra: float | None
    dec: float | None
    snr: float
    band: str
    survey: str
    psf_flux: float
    psf_flux_err: float
    ml: float
    ml_version: str

    @computed_field()
    @property
    def isdiffpos(self) -> bool:
        """
        Determine if the observation is a positive detection

        :return: True if positive detection, False otherwise
        """
        return (self.psf_flux > 0.0) & (pd.notnull(self.magpsf))

    @computed_field()
    @property
    def mjd(self) -> float:
        """
        Convert JD to MJD
        """
        return self.jd - 2400000.5


def convert_photometry(
    df: pd.DataFrame,
    extra_df: pd.DataFrame,
) -> list[Observation]:
    """
    Convert photometry dataframes to list of Observation objects

    :param df: DataFrame of main photometry
    :param extra_df: DataFrame of extra photometry
    :return: List of Observation objects
    """
    cols = [x for x in df.columns if x in extra_df.columns]
    combo_df = (
        pd.concat([df[cols], extra_df[cols].astype(df[cols].dtypes)])
        if not extra_df.empty
        else df
    )

    combo_df = (
        combo_df.sort_values(by="jd")
        .reset_index(drop=True)
        .rename(
            columns={
                "psfFlux": "psf_flux",
                "psfFluxErr": "psf_flux_err",
            }
        )
    ).replace({None: np.nan})

    return [Observation(**row) for row in combo_df.to_dict(orient="records")]


def get_extra_df(alert) -> pd.DataFrame:
    """
    Get extra photometry dataframe from alert survey matches

    :param alert: Alert object
    :return: DataFrame of extra photometry
    """

    ztf_df = pd.DataFrame()
    if hasattr(alert.survey_matches, "ztf"):
        if alert.survey_matches.ztf is not None:
            # try:
            #     match = get_object(
            #         "ZTF",
            #         object_id=alert.survey_matches.ztf.objectId,
            #     )
            #
            #     ztf_df = pd.DataFrame(
            #         x.model_dump() for x in match.get_photometry()
            #     )
            #     ztf_df["survey"] = "ztf"
            #
            # except APINotFoundError:
            #     logger.error(f"No ZTF match found for alert {alert.objectId} / {alert.survey_matches.ztf.objectId}")
            #
            # except ValidationError:
            #     logger.error(f"Validation error for ZTF match photometry for alert {alert.objectId} / {alert.survey_matches.ztf.objectId}")

            try:

                res = download_kowalski_alert_data(
                    source_name=alert.survey_matches.ztf.objectId
                )
                ztf_df = pd.DataFrame([res[0]["candidate"]] + res[0]["prv_candidates"])

                ztf_df["band"] = ztf_df["fid"].map({1: "g", 2: "r", 3: "i"})

                # Flux in nanojanskys
                flux = (ztf_df["magpsf"].to_numpy() * u.ABmag).to(u.nanojansky).value

                # Flux error in nanojanskys,
                # using the magnitude error to calculate the flux error
                flux_err = 0.5 * (
                    ((ztf_df["magpsf"] - ztf_df["sigmapsf"]).to_numpy() * u.ABmag)
                    .to(u.nanojansky)
                    .value
                    - ((ztf_df["magpsf"] + ztf_df["sigmapsf"]).to_numpy() * u.ABmag)
                    .to(u.nanojansky)
                    .value
                )

                ztf_df["psfFlux"] = flux
                ztf_df["psfFluxErr"] = flux_err

                ztf_df["snr"] = ztf_df["psfFlux"] / ztf_df["psfFluxErr"]

                ztf_df["survey"] = "ztf"

            except ValueError:
                logger.error(
                    f"Value error for ZTF match photometry for alert {alert.objectId} / {alert.survey_matches.ztf.objectId}"
                )

    lsst_df = pd.DataFrame()
    if hasattr(alert.survey_matches, "lsst"):
        if alert.survey_matches.lsst is not None:

            match = get_object(
                "LSST",
                object_id=alert.survey_matches.lsst.objectId,
            )

            lsst_df = pd.DataFrame(x.model_dump() for x in match.get_photometry())
            lsst_df["survey"] = "lsst"

    return pd.concat([ztf_df, lsst_df], ignore_index=True)


class Source(BaseModel):
    objectid: int | str
    jd: float
    ztfid: str | None
    lsstid: str | None
    ra: float
    dec: float

    photometry: list[Observation]

    @computed_field
    @property
    def ndethist(self) -> int:
        """
        Get number of positive detections

        :return: Number of positive detections
        """
        return len(self.get_detections())

    @computed_field
    @property
    def filters(self) -> list[str]:
        """
        Get list of unique filters in detections

        :return: List of unique filters
        """
        return list(set(self.get_detections()["band"]))

    @computed_field
    @property
    def ndetfilters(self) -> int:
        """
        Get number of unique filters in detections

        :return: Number of unique filters
        """
        return len(self.filters)

    @computed_field
    @property
    def jdstarthist(self) -> float:
        """
        Get JD of first positive detection

        :return: JD of first positive detection
        """
        return self.get_detections()["jd"].min()

    @computed_field
    @property
    def jdendhist(self) -> float:
        """
        Get JD of last positive detection

        :return: JD of last positive detection
        """
        return self.get_detections()["jd"].max()

    @computed_field
    @property
    def age(self) -> float:
        """
        Get age of source in days (time between first and last positive detection)

        :return: Age of source in days
        """
        return self.jdendhist - self.jdstarthist

    def get_photometry(self) -> pd.DataFrame:
        """
        Get photometry as DataFrame

        :return: DataFrame of photometry
        """
        return pd.DataFrame([x.model_dump() for x in self.photometry])

    def get_detections(self) -> pd.DataFrame:
        """
        Get positive detections from photometry

        :return: DataFrame of positive detections
        """
        df = self.get_photometry()
        positive_det_mask = (df["isdiffpos"] == True) & (pd.notnull(df["magpsf"]))
        return df[positive_det_mask].reset_index(drop=True)

    @classmethod
    def from_lsst(cls, lsst_id: int | str) -> "Source":
        """
        Create Source object from raw LSST alert

        :param lsst_id: LSST object ID
        :return: Source object
        """

        alert = get_object(
            "LSST",
            object_id=str(lsst_id),
        )

        photometry = [x.model_dump() for x in alert.get_photometry()]

        lsst_df = pd.DataFrame(photometry)
        lsst_df["survey"] = "lsst"

        extra_df = get_extra_df(alert)

        photometry = convert_photometry(lsst_df, extra_df)

        return cls(
            objectid=alert.objectId,
            lsstid=alert.objectId,
            ztfid=(
                alert.survey_matches.ztf.objectId
                if alert.survey_matches.ztf is not None
                else None
            ),
            photometry=photometry,
            **alert.candidate.model_dump(),
        )

    @classmethod
    def from_ztf(cls, raw_alert) -> "Source":
        """
        Create Source object from raw ZTF alert

        :param raw_alert: Raw alert data
        :return: Source object
        """
        alert = ZtfAlert(**raw_alert)

        photometry = [x.model_dump() for x in alert.get_photometry()]
        ztf_df = pd.DataFrame(photometry)
        ztf_df["survey"] = "ztf"

        extra_df = get_extra_df(alert)

        photometry = convert_photometry(ztf_df, extra_df)

        return cls(
            objectid=alert.objectId,
            lsstid=(
                alert.survey_matches.lsst.objectId
                if alert.survey_matches.lsst is not None
                else None
            ),
            ztfid=alert.objectId,
            photometry=photometry,
            **alert.candidate.model_dump(),
        )

    @classmethod
    def from_alert(cls, raw_alert, category: str) -> "Source":
        """
        Create Source object from raw alert based on category

        :param raw_alert: Raw alert data
        :param category: Category of alert ("lsst" or "ztf")
        :return: Source object
        """
        if category == "lsst":
            return cls.from_lsst(raw_alert)
        elif category == "ztf":
            return cls.from_ztf(raw_alert)
        else:
            err = f"Unrecognised category {category}"
            logger.error(err)
            raise ValueError(err)

    def to_archive(self):
        """
        Dump Source object to dictionary in archive format,
        with photometry as list of dictionaries

        :return: Dictionary in archive format
        """

        output_path = babamul_cache_dir / f"{self.objectid}.json"
        res = self.model_dump()
        with open(output_path, "w") as f:
            json.dump(res, f, indent=4)

    def convert_to_ztfstyle(self) -> dict:
        res = self.model_dump()
        cand = {key: val for key, val in res.items() if key not in ["photometry"]}
        res["candidate"] = cand
        res["objectId"] = self.objectid

        df = self.get_detections()

        df["fid"] = df["band"].map(FID_MAPPING).astype("Int8")
        df["filter"] = df["band"]

        match = df.iloc[-1]

        res["candidate"].update(match.to_dict())
        res["prv_candidates"] = df[:-1].to_dict(orient="records")

        return res
