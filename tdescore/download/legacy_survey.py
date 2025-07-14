"""
Code to download legacy survey data from the DESI Legacy Imaging Surveys
"""
import json

from dl import queryClient as qc
import pandas as pd
from pathlib import Path
from tdescore.paths import legacy_survey_dir
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


LEGACY_SURVEY_DR = "DR10"

default_catalog = f'ls_{LEGACY_SURVEY_DR.lower()}'

def legacy_survey_path(source_name: str, data_release = LEGACY_SURVEY_DR) -> Path:
    """
    Get path to Legacy survey json cache

    :param source_name: Name of source
    :param data_release: Data release (default is DR10)
    :return: path
    """
    dr_dir = legacy_survey_dir / data_release
    dr_dir.mkdir(exist_ok=True)
    return dr_dir / f"{source_name}.json"

def get_ls_redshift(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 3.,
    catalog=default_catalog,
) -> pd.Series:
    """
    Function to query the DESI Legacy Imaging Surveys database for photo-z data

    :param ra_deg: RA in degrees
    :param dec_deg: Dec in degrees
    :param radius_arcsec: Search radius in arcseconds
    :param catalog: Catalog name (default is 'ls_dr10')

    :return: DataFrame with the first match
    """
    radius_deg = radius_arcsec / 3600.
    res = qc.query(
        sql=f"SELECT z_spec, z_phot_median, z_phot_std, z_phot_l95, ra, dec, type, flux_z from {catalog}.photo_z INNER JOIN {catalog}.tractor ON {catalog}.tractor.ls_id = {catalog}.photo_z.ls_id where 't' = Q3C_RADIAL_QUERY(ra, dec, {ra_deg}, {dec_deg}, {radius_deg}) LIMIT 1",
        fmt='pandas'
    )

    if len(res) == 0:
        return pd.Series()

    return res.iloc[0]


def download_legacy_survey_data(
    src_table: pd.DataFrame,
    data_release: str = LEGACY_SURVEY_DR,
    search_radius: float = 1.5,
):
    """
    Function to download Legacy Survey crossmatch data for each source in a dataframe,
    and save each as a json file

    :param src_table: table of sources
    :param data_release: Data release (default is DR10)
    :param search_radius: search radius in arcsec
    :return: None
    """

    logger.info("Downloading Legacy Survey data")

    for _, row in tqdm(src_table.iterrows(), total=len(src_table)):
        output_path = legacy_survey_path(row["ztf_name"], data_release=data_release)

        if not output_path.exists():

            catalog = f"ls_{data_release.lower()}"

            try:
                # pylint: disable=no-member
                catalog_data = get_ls_redshift(
                    ra_deg=row["ra"],
                    dec_deg=row["dec"],
                    radius_arcsec=search_radius,
                    catalog=catalog,
                )

                rename = {x: f"{catalog}_{x}" for x in catalog_data.index}
                catalog_data.rename(rename, inplace=True)

                if len(catalog_data) == 0:
                    res = json.dumps({})
                else:
                    res = catalog_data.to_json()

                with open(output_path, "w", encoding="utf8") as out_f:
                    out_f.write(res)

            except KeyError:
                err = f"Failed to download {data_release} data for {row['ztf_name']}"
                logger.error(err)
