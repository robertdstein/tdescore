"""
Module for plotting variable distributions
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from tdescore.classifications.tde import is_tde
from tdescore.combine.parse import load_metadata
from tdescore.paths import features_dir


def plot_all_histograms(column: str, metadata: pd.DataFrame):
    """
    Plots the distribution of a variable,
    split into separate histograms for each subclass

    :param column: column name (i.e variable)
    :param metadata: dataframe containing aggregated metadata
    :return: None
    """
    data = metadata[pd.notnull(metadata[column])]

    raw_classes = ["Tidal Disruption Event"] + data["fritz_class"].tolist()
    classes = sorted(set(raw_classes), key=raw_classes.index)

    _, axes = plt.subplots(
        len(classes), 1, figsize=(5, 3.0 + 2.0 * len(classes)), sharex=True
    )

    for i, class_name in enumerate(classes):
        sources = data[data["fritz_class"].to_numpy("str") == str(class_name)]
        axes[i].hist(
            sources[column],
            range=(min(data[column]), max(data[column])),
            bins=50,
            color=f"C{i}",
        )
        axes[i].set_title(f"class = {class_name}")

    out_path = features_dir.joinpath(f"all_classes_{column}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_pair_histograms(column: str, metadata: pd.DataFrame):
    """
    Plots the distribution of a variable in one figure,
    with one normalised distributions for 'TDE' and another for 'non-TDE'

    :param column: column name (i.e variable)
    :param metadata: dataframe containing aggregated metadata
    :return: None
    """

    mask = is_tde(metadata["ztf_name"])

    tde = metadata[mask][column]
    tde_nan = pd.notnull(tde)
    f_nan_tde = np.mean(tde_nan)
    tde_w = np.ones_like(tde[tde_nan]) / np.sum(tde_nan)

    bkg = metadata[~mask][column]
    bkg_nan = pd.notnull(bkg)
    f_nan_bkg = np.mean(bkg_nan)
    bkg_w = np.ones_like(bkg[bkg_nan]) / np.sum(bkg_nan)

    plt.figure()
    plt.title(column)
    kwargs = {
        "range": (
            min(tde[tde_nan].tolist() + bkg[bkg_nan].tolist()),
            max(tde[tde_nan].tolist() + bkg[bkg_nan].tolist()),
        ),
        "bins": 50,
        "alpha": 0.5,
    }

    plt.hist(
        tde[tde_nan],
        weights=tde_w,
        color="green",
        label=f"TDE ({100. * f_nan_tde:.0f}%)",
        **kwargs,
        zorder=5,
    )
    plt.hist(
        bkg[bkg_nan],
        weights=bkg_w,
        color="red",
        label=f"Non-TDE ({100. * f_nan_bkg:.0f}%)",
        **kwargs,
    )
    plt.legend()

    out_path = features_dir.joinpath(f"pair_{column}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def batch_plot_variables(metadata: Optional[pd.DataFrame] = None):
    """
    Iteratively plot distributions for each column

    :return: None
    """
    if metadata is None:
        metadata = load_metadata()

    print(metadata.columns)

    # pylint: disable=E1101
    cols = [
        x for i, x in enumerate(metadata.columns) if not metadata.dtypes[i] == object
    ]
    for col in tqdm(cols):
        plot_all_histograms(column=col, metadata=metadata)
        plot_pair_histograms(column=col, metadata=metadata)
