"""
Central module for handling paths of directories
"""
import os
from pathlib import Path

ranking_dir = os.environ.get("TDE_RANKING_DIR", None)

if ranking_dir is None:
    raise ValueError(
        "Must specify a directory for ZTF ranking data. "
        "Set the TDE_RANKING_DIR environment variable."
    )

ranking_dir = Path(ranking_dir)

alert_dir = ranking_dir.parent.joinpath("alerts")

data_dir = Path(os.getenv("TDESCORE_DATA", ""))
data_dir.mkdir(exist_ok=True)

ampel_cache_dir = data_dir.joinpath("ampel")
ampel_cache_dir.mkdir(exist_ok=True)

gaia_cache_dir = data_dir.joinpath("gaia")
gaia_cache_dir.mkdir(exist_ok=True)
panstarrs_cache_dir = data_dir.joinpath("panstarrs")
panstarrs_cache_dir.mkdir(exist_ok=True)

lightcurve_dir = data_dir.joinpath("lightcurve_plots")
lightcurve_dir.mkdir(exist_ok=True)

lightcurve_metadata_dir = data_dir.joinpath("lightcurve_metadata")
lightcurve_metadata_dir.mkdir(exist_ok=True)

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
