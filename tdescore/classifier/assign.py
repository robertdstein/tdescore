"""
Script to collate classifications from TNS, BTS, Fritz, Growth, SDSS, Gaia,
WISE, Milliquas, and Gaia QSO candidates
"""
import logging

import numpy as np
import pandas as pd

from tdescore.classifications.mapping import bts_map, fritz_map, growth_map, tns_map
from tdescore.raw.tde import is_tde

logger = logging.getLogger(__name__)


def assign_classification_origin(
    sources: pd.DataFrame,
    non_spectra_marshal_classes: bool = False,
) -> pd.DataFrame:
    """
    Get sources with classifications from TNS, BTS, Fritz, Growth, SDSS, Gaia,
    WISE, Milliquas, and Gaia QSO candidates

    :param sources: Sources to use. If None, use all classified sources
    :param non_spectra_marshal_classes: Whether to include non-spectra marshal classes
    :return: Sources with classifications that can be attributed to an origin
    """

    tde_bool = is_tde(sources["ztf_name"])

    base_class = np.array([tns_map[x] for x in sources["tns_classification"]])
    base_class[~tde_bool & (base_class == "TDE")] = None
    base_class[tde_bool & (base_class != "TDE")] = None

    sources["subclass"] = base_class
    sources["class_origin"] = None

    sources.loc[~pd.isnull(sources["subclass"]), ["class_origin"]] = "TNS"

    has_spectra = sources["fritz_spectrum_exists"] + sources["growth_n_spectra"] > 0.0
    sources["has_spectrum"] = has_spectra

    # BTS
    bts_class = np.array([bts_map[x] for x in sources["bts_class"]])
    bts_crossmatch_class = np.array(
        [bts_map[x] for x in sources["crossmatch_bts_class"]]
    )

    differing = (
        (bts_class != bts_crossmatch_class)
        & ~(pd.isnull(sources["bts_class"]))
        & ~(pd.isnull(sources["crossmatch_bts_class"]))
    )

    assert np.sum(differing) == 0

    # Fill missing from BTS
    missing = pd.isnull(bts_class)
    bts_class[missing] = bts_crossmatch_class[missing]

    for replace_mask in [
        ~(pd.isnull(sources["subclass"])),
        (bts_class == "bogus") & ~(pd.isnull(sources["subclass"])),
    ]:
        bts_class[replace_mask] = sources["subclass"][replace_mask]

    bts_class[~tde_bool & (bts_class == "TDE")] = None
    bts_class[tde_bool & (bts_class != "TDE")] = None

    # If there is nothing in TNS, take BTS classification
    missing = (pd.isnull(sources["subclass"])) & ~(pd.isnull(bts_class)) & has_spectra
    sources.loc[missing, ["subclass"]] = bts_class[missing]
    sources.loc[missing, ["class_origin"]] = "BTS-spec"

    differing = (
        (bts_class != sources["subclass"])
        & ~(pd.isnull(bts_class))
        & ~(pd.isnull(sources["subclass"]))
    )
    assert np.sum(differing) == 0, "BTS and TNS classifications differ"

    # Fritz
    base_fritz_classes = [
        x.split(",") if x is not None else None for x in sources["fritz_current_class"]
    ]

    parsed_fritz_class = []

    for i, row in enumerate(base_fritz_classes):
        new = []
        if (row is not None) & (row != [""]):
            for source_class in row:
                if source_class not in [
                    "nuclear",
                    "hosted",
                    "Galactic Nuclei",
                    "transient",
                ]:
                    if source_class not in fritz_map:
                        print(f"new: {source_class} ({row})")
                    else:
                        new.append(fritz_map[source_class])

                # Sort and go for latest

                new = sorted(set(new), key=new[::-1].index)

                # Defer to the latest human classification
                if len(new) > 1:
                    if sources["subclass"].iloc[i] in new:
                        new = [sources["subclass"].iloc[i]]

                    if not sources["has_spectrum"].iloc[i]:
                        new = []

                    if len(new) > 1:
                        new = [x for x in new][-1:]

        if len(new) == 1:
            parsed_fritz_class.append(new[0])
        else:
            parsed_fritz_class.append(None)

    parsed_fritz_class = np.array(parsed_fritz_class)
    parsed_fritz_class[~tde_bool & (parsed_fritz_class == "TDE")] = None
    parsed_fritz_class[tde_bool & (parsed_fritz_class != "TDE")] = None

    unclassified = (
        (
            np.array(
                [
                    x in ["SN", None, "Galaxy", "Other", "bogus"]
                    for x in sources["subclass"]
                ]
            )
        )
        & ~(pd.isnull(parsed_fritz_class))
        & has_spectra
    )

    sources.loc[unclassified, ["subclass"]] = parsed_fritz_class[unclassified]
    sources.loc[unclassified, ["class_origin"]] = "Fritz-spec"

    growth_class = np.array([growth_map[x] for x in sources["growth_class"]])
    growth_class[~tde_bool & (growth_class == "TDE")] = None
    growth_class[tde_bool & (growth_class != "TDE")] = None

    # If there is nothing yet, take Growth classification
    missing = (
        (pd.isnull(sources["subclass"])) & ~(pd.isnull(growth_class)) & has_spectra
    )
    sources.loc[missing, ["subclass"]] = growth_class[missing]
    sources.loc[missing, ["class_origin"]] = "Growth-spec"

    # Finally, add TDEs
    sources.loc[pd.isnull(sources["subclass"]) & tde_bool, ["class_origin"]] = "TDE"
    sources.loc[pd.isnull(sources["subclass"]) & tde_bool, ["subclass"]] = "TDE"

    # Non-spectra

    # SDSS classifications
    sources.loc[
        (pd.isnull(sources["subclass"])) & (sources["sdss_class"] == "STAR"),
        ["subclass", "class_origin"],
    ] = ["Varstar", "SDSS-spec"]
    sources.loc[
        (pd.isnull(sources["subclass"])) & (sources["sdss_class"] == "QSO"),
        ["subclass", "class_origin"],
    ] = ["AGN", "SDSS-spec"]
    sources.loc[
        (pd.isnull(sources["subclass"]))
        & (sources["sdss_class"] == "GALAXY")
        & sources["sdss_subclass"].isin(["AGN", "AGN BROADLINE"]),
        ["subclass", "class_origin"],
    ] = ["AGN", "SDSS-spec"]

    # Gaia parallax
    # sources.loc[
    #     (pd.isnull(sources["subclass"])) & (sources["gaia_aplx"] > 5.0),
    #     ["subclass", "class_origin"]
    # ] = ["Varstar", "Gaia-Parallax"]

    # Probable AGN
    sources.loc[
        (pd.isnull(sources["subclass"])) & (sources["has_milliquas"] > 0.8),
        ["subclass", "class_origin"],
    ] = ["AGN", "Milliquas"]
    sources.loc[
        (pd.isnull(sources["subclass"]))
        & ~(pd.isnull(sources["w1_m_w2"]))
        & (sources["w1_m_w2"] > 0.8),
        ["subclass", "class_origin"],
    ] = ["AGN", "WISE"]
    sources.loc[
        (pd.isnull(sources["subclass"])) & (sources["gaia_in_qso_candidates"] > 0.5),
        ["subclass", "class_origin"],
    ] = ["AGN", "Gaia-QSO"]

    if non_spectra_marshal_classes:
        sources.loc[
            (pd.isnull(sources["subclass"]))
            & (sources["phot_variable_flag"] == "VARIABLE"),
            ["subclass", "class_origin"],
        ] = ["Variable", "Gaia-Variable"]

        # finally Fritz no spectra
        unclassified = (
            pd.isnull(sources["subclass"])
            & ~has_spectra
            & ~(pd.isnull(parsed_fritz_class))
        )
        sources.loc[unclassified, ["subclass"]] = parsed_fritz_class[unclassified]
        sources.loc[unclassified, ["class_origin"]] = "Fritz-other"

        # If there is nothing yet, take Growth classification
        unclassified = (
            pd.isnull(sources["subclass"]) & ~has_spectra & ~(pd.isnull(growth_class))
        )
        sources.loc[unclassified, ["subclass"]] = growth_class[unclassified]
        sources.loc[unclassified, ["class_origin"]] = "Growth-other"

        # If there is nothing yet, take Growth classification
        unclassified = (
            pd.isnull(sources["subclass"]) & ~has_spectra & ~(pd.isnull(bts_class))
        )
        sources.loc[unclassified, ["subclass"]] = bts_class[unclassified]
        sources.loc[unclassified, ["class_origin"]] = "BTS-other"

    return sources
