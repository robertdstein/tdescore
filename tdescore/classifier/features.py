"""
Module containing the list of all features used in the classifier
"""
import numpy as np

wise_columns = [
    ("w1_m_w2", r"WISE W1$-$W2 host colour"),
    ("w3_m_w4", "WISE W3$-$W4 host colour"),
    ("w1_chi2", r"WISE W1 $\chi^{2}$"),
]

fast_host_columns = [
    ("sgscore1", "Star/Galaxy Score for PS1 host"),
    ("has_milliquas", "Has milliquas crossmatch?"),
    ("g-r_MeanPSFMag", r"PS1 host $g-r$ colour"),
    ("r-i_MeanPSFMag", r"PS1 host $r-i$ colour"),
    ("i-z_MeanPSFMag", r"PS1 host $i-z$ colour"),
    ("z-y_MeanPSFMag", r"PS1 host $z-y$ colour"),
    ("strm_class", "PS1strm host class"),
    ("strm_prob_Galaxy", "PS1strm host prob. galaxy"),
    ("strm_prob_Star", "PS1strm host prob. star"),
    ("strm_prob_QSO", "PS1strm host prob. QSO"),
    ("gaia_aplx", "Absolute Gaia parallax"),
]

host_columns = wise_columns + fast_host_columns

early_base = host_columns + [("distpsnr1", "Distance to PS1 host")]

early_to_use = ["sharpnr", "sigmapsf", "classtar", "scorr", "magdiff"]

early_columns = early_base + [
    (f"early_{x}", f"Median {x} in 24hours") for x in early_to_use
]


peak_columns = (
    early_base
    + [
        ("peak_color", "Colour at g-band peak"),
        ("pre_inflection", "Number of pre-peak inflections"),
        ("positive_fraction", "Fraction of positive detections"),
        ("det_cadence", "Mean detection candence"),
        ("y_scale", "Y Scale from G.P."),
    ]
    + [
        ("classtar", r"\texttt{SourceExtractor} variable"),
        ("sumrat", "`Sum ratio'"),
        ("distnr", "Pixel distance to nearest source"),
        ("high_noise", "High white noise fitted by G.P."),
    ]
)

post_peak = peak_columns + [
    ("color_grad", "Rate of colour change"),
    ("fade", "Fade from G.P."),
    ("length_scale", "Length scale from G.P."),
    ("post_inflection", "Number of post-peak inflections"),
    ("score", "Score from G.P"),
    ("sncosmo_chisq", r"sncosmo $\chi^{2}$"),
    ("sncosmo_chi2pdof", r"sncosmo $\chi^{2}$ per d.o.f"),
    ("sncosmo_x1", "sncosmo X1 parameter"),
    ("sncosmo_c", "sncosmo c parameter"),
]


def parse_columns(columns: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """
    Function to parse a list of columns into a list of column names

    :param columns: list of columns
    :return: list of column names, list of column descriptions
    """
    relevant_columns = list(np.array(columns).T[0])
    column_descriptions = list(np.array(columns).T[1])
    return relevant_columns, column_descriptions


default_columns, default_descriptions = parse_columns(post_peak)
