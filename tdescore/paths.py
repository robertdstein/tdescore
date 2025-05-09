"""
Central module for handling paths of directories
"""
import os
from pathlib import Path

data_dir = Path(os.getenv("TDESCORE_DATA", ""))
data_dir.mkdir(exist_ok=True)

ampel_cache_dir = data_dir.joinpath("ampel")
ampel_cache_dir.mkdir(exist_ok=True)

kowalski_cache_dir = data_dir.joinpath("kowalski")
kowalski_cache_dir.mkdir(exist_ok=True)

gaia_cache_dir = data_dir.joinpath("gaia")
gaia_cache_dir.mkdir(exist_ok=True)
panstarrs_cache_dir = data_dir.joinpath("panstarrs")
panstarrs_cache_dir.mkdir(exist_ok=True)
ps1strm_cache_dir = data_dir.joinpath("ps1strm")
ps1strm_cache_dir.mkdir(exist_ok=True)
wise_cache_dir = data_dir.joinpath("wise")
wise_cache_dir.mkdir(exist_ok=True)
sdss_cache_dir = data_dir.joinpath("sdss")
sdss_cache_dir.mkdir(exist_ok=True)
fritz_cache_dir = data_dir.joinpath("fritz")
fritz_cache_dir.mkdir(exist_ok=True)
tns_cache_dir = data_dir.joinpath("tns")
tns_cache_dir.mkdir(exist_ok=True)
legacy_survey_dir = data_dir.joinpath("legacy_survey")
legacy_survey_dir.mkdir(exist_ok=True)

lightcurve_dir = data_dir.joinpath("lightcurve_plots")
lightcurve_dir.mkdir(exist_ok=True)

lightcurve_metadata_dir = data_dir.joinpath("lightcurve_metadata")
lightcurve_metadata_dir.mkdir(exist_ok=True)
lightcurve_infant_dir = data_dir.joinpath("lightcurve_infant")
lightcurve_infant_dir.mkdir(exist_ok=True)
lightcurve_week_dir = data_dir.joinpath("lightcurve_week")
lightcurve_week_dir.mkdir(exist_ok=True)
lightcurve_month_dir = data_dir.joinpath("lightcurve_month")
lightcurve_month_dir.mkdir(exist_ok=True)
lightcurve_thermal_dir = data_dir.joinpath("lightcurve_thermal")
lightcurve_thermal_dir.mkdir(exist_ok=True)
lightcurve_resampled_dir = data_dir.joinpath("lightcurve_resampled")
lightcurve_resampled_dir.mkdir(exist_ok=True)

sncosmo_dir = data_dir.joinpath("sncosmo")
sncosmo_dir.mkdir(exist_ok=True)
sn_cosmo_plot_dir = data_dir.joinpath("sncosmo_plots")
sn_cosmo_plot_dir.mkdir(exist_ok=True)

combined_metadata_path = data_dir.joinpath("combined_metadata.json")

features_dir = data_dir.joinpath("features")
features_dir.mkdir(exist_ok=True)

tde_list = data_dir.joinpath("ZTF-II TDEs - Current TDEs.csv")

combined_ranking_file = data_dir.joinpath("all_ranking_data.csv")
summary_ranking_file = data_dir.joinpath("ranking_data.csv")
summary_alert_file = data_dir.joinpath("alert_data.csv")

sfd_path = data_dir.joinpath("sfdmap/sfddata-master")
