"""
Module containing the list of all features used in the classifier
"""
import numpy as np

host_columns = [
    ("sgscore1", "Star/Galaxy Score for PS1 host"),
    ("w1_m_w2", r"WISE W1$-$W2 host colour"),
    ("w3_m_w4", "WISE W3$-$W4 host colour"),
    ("w1_chi2", r"WISE W1 $\chi^{2}$"),
    ("has_milliquas", "Has milliquas crossmatch?"),
    ("g-r_MeanPSFMag", r"PS1 host $g-r$ colour"),
    ("r-i_MeanPSFMag", r"PS1 host $r-i$ colour"),
    ("i-z_MeanPSFMag", r"PS1 host $i-z$ colour"),
    ("z-y_MeanPSFMag", r"PS1 host $z-y$ colour"),
]

early_columns = host_columns + [
    ("distpsnr1", "Distance to PS1 host"),
    ("classtar", r"\texttt{SourceExtractor} variable"),
    ("sumrat", "`Sum ratio'"),
    ("distnr", "Pixel distance to nearest source"),
]

peak_columns = early_columns + [
    ("peak_color", "Colour at g-band peak"),
    ("pre_inflection", "Number of pre-peak inflections"),
    ("positive_fraction", "Fraction of positive detections"),
    ("det_cadence", "Mean detection candence"),
    ("y_scale", "Y Scale from G.P."),
]

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
