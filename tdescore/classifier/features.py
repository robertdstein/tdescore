"""
Module containing the list of all features used in the classifier
"""
import numpy as np

host_columns = [
    ("distpsnr1", "Distance to PS1 host"),
    ("sgscore1", "Star/Galaxy Score for PS1 host"),
    ("w1_m_w2", "WISE W1-W2 host colour"),
    ("w3_m_w4", "WISE W3-W4 host colour"),
    ("w1_chi2", r"WISE W1 $\chi^{2}$"),
    ("has_milliquas", "Has milliquas crossmatch?"),
    ("g-r_MeanPSFMag", "PS1 host g-r colour"),
    ("r-i_MeanPSFMag", "PS1 host r-i colour"),
    ("i-z_MeanPSFMag", "PS1 host i-z colour"),
    ("z-y_MeanPSFMag", "PS1 host z-y colour"),
]

peak_columns = host_columns + [
    ("color_grad", "Rate of colour change"),
    ("peak_color", "Colour at g-band peak"),
    ("pre_inflection", "Number of pre-peak inflections"),
    ("positive_fraction", "Fraction of positive detections"),
    ("det_cadence", "Mean detection candence"),
    ("length_scale", "Length scale from G.P."),
    ("y_scale", "Y Scale from G.P."),
    ("classtar", r"\texttt{SourceExtractor} variable"),
    ("sumrat", "`Sum ratio'"),
    ("distnr", "Pixel distance to nearest source"),
]

post_peak = peak_columns + [
    ("fade", "Fade from G.P."),
    ("post_inflection", "Number of post-peak inflections"),
    ("score", "Score from G.P"),
    ("sncosmo_chisq", r"sncosmo $\chi^{2}$"),
    ("sncosmo_chi2pdof", r"sncosmo $\chi^{2}$ per d.o.f"),
    ("sncosmo_x1", "sncosmo X1 parameter"),
    ("sncosmo_c", "sncosmo c parameter"),
]

relevant_columns = list(np.array(post_peak).T[0])
column_descriptions = list(np.array(post_peak).T[1])
