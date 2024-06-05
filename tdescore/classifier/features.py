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
    # ("strm_class", "PS1strm host class"),
    ("strm_prob_Galaxy", "PS1strm host prob. galaxy"),
    ("strm_prob_Star", "PS1strm host prob. star"),
    ("strm_prob_QSO", "PS1strm host prob. QSO"),
    ("gaia_aplx", "Absolute Gaia parallax"),
]

host_columns = wise_columns + fast_host_columns

infant_base_cols = host_columns + [
    ("infant_n_detections", "infant_n_detections"),
    ("infant_has_g", "infant_has_g"),
    ("infant_has_r", "infant_has_r"),
]

infant_columns = infant_base_cols + [
    ("infant_rb", "infant_rb"),
    ("infant_distnr", "infant_distnr"),
    ("infant_magdiff", "infant_magdiff"),
    ("infant_sigmapsf", "infant_sigmapsf"),
    ("infant_chipsf", "infant_chipsf"),
    ("infant_sumrat", "infant_sumrat"),
    ("infant_fwhm", "infant_fwhm"),
    ("infant_elong", "infant_elong"),
    ("infant_chinr", "infant_chinr"),
    ("infant_sharpnr", "infant_sharpnr"),
    ("infant_scorr", "infant_scorr"),
    ("infant_offset_med", "infant_offset_med"),
    ("infant_ul_delay", "infant_ul_delay"),
    ("infant_ul_rise", "infant_ul_rise"),
    ("infant_ul_grad", "infant_ul_grad"),
]

week_base_cols = infant_base_cols + [
    ("week_g_rise", "week_g_rise"),
    ("week_r_rise", "week_r_rise"),
    ("week_median_color", "week_median_color"),
]

week_columns = week_base_cols + [
    ("week_rb", "week_rb"),
    ("week_distnr", "week_distnr"),
    ("week_magdiff", "week_magdiff"),
    ("week_sigmapsf", "week_sigmapsf"),
    ("week_chipsf", "week_chipsf"),
    ("week_sumrat", "week_sumrat"),
    ("week_fwhm", "week_fwhm"),
    ("week_elong", "week_elong"),
    ("week_chinr", "week_chinr"),
    ("week_sharpnr", "week_sharpnr"),
    ("week_scorr", "week_scorr"),
    ("week_offset_med", "week_offset_med"),
    ("week_n_detections", "week_n_detections"),
]

month_columns = week_base_cols + [
    ("month_rise", "month_rise"),
    ("month_intercept", "month_intercept"),
    ("month_color", "month_color"),
    ("month_chi2", "month_chi2"),
    ("mean_month_chi2", "mean_month_chi2"),
    ("month_rb", "month_rb"),
    ("month_distnr", "month_distnr"),
    ("month_magdiff", "month_magdiff"),
    ("month_sigmapsf", "month_sigmapsf"),
    ("month_chipsf", "month_chipsf"),
    ("month_sumrat", "month_sumrat"),
    ("month_fwhm", "month_fwhm"),
    ("month_elong", "month_elong"),
    ("month_chinr", "month_chinr"),
    ("month_sharpnr", "month_sharpnr"),
    ("month_scorr", "month_scorr"),
    ("month_offset_med", "month_offset_med"),
    ("month_n_detections", "month_n_detections"),
]

base_thermal_columns = (
    week_base_cols
    + [
        ("month_rise_padded", "month_rise"),
        ("month_intercept_padded", "month_intercept"),
        ("month_color_padded", "month_color"),
        ("mean_month_chi2_padded", "mean_month_chi2"),
        ("distpsnr1", "Distance to nearest PS1 source"),
        # ("fade", "Fade from G.P."),
        # ("peak_color", "Colour at g-band peak"),
        ("positive_fraction", "Fraction of positive detections"),
    ]
    + [
        ("thermal_offset_med", "thermal_offset_med"),
        ("thermal_log_temp_peak", "thermal_log_temp_peak"),
        ("thermal_log_temp_sigma", "thermal_log_temp_sigma"),
        ("thermal_cooling", "thermal_cooling"),
        ("thermal_cooling_sigma", "thermal_cooling_sigma"),
        ("thermal_log_temp_ll", "thermal_log_temp_ll"),
        ("thermal_log_temp_ul", "thermal_log_temp_ul"),
        ("thermal_cooling_ll", "thermal_cooling_ll"),
        ("thermal_cooling_ul", "thermal_cooling_ul"),
        ("thermal_score", "thermal_score"),
        ("thermal_length_scale", "thermal_length_scale"),
        ("thermal_y_scale", "thermal_y_scale"),
    ]
)

thermal_columns = base_thermal_columns + [
    ("thermal_distnr", "thermal_distnr"),
    ("thermal_sigmapsf", "thermal_sigmapsf"),
    # ('thermal_chipsf', 'thermal_chipsf'),
    ("thermal_sumrat", "thermal_sumrat"),
    ("thermal_fwhm", "thermal_fwhm"),
    ("thermal_sharpnr", "thermal_sharpnr"),
    ("thermal_post_inflection", "thermal_post_inflection"),
    ("thermal_det_cadence", "thermal_det_cadence"),
]

post_peak = (
    base_thermal_columns
    + [
        ("peak_color", "Colour at g-band peak"),
        ("det_cadence", "Mean detection candence"),
        ("y_scale", "Y Scale from G.P."),
        ("color_grad", "Rate of colour change"),
        ("pre_inflection", "Number of pre-peak inflections"),
        ("sncosmo_chisq", r"sncosmo $\chi^{2}$"),
        ("sncosmo_chi2pdof", r"sncosmo $\chi^{2}$ per d.o.f"),
        ("sncosmo_x1", "sncosmo X1 parameter"),
        ("sncosmo_c", "sncosmo c parameter"),
    ]
    + [
        ("classtar", r"\texttt{SourceExtractor} variable"),
        ("sumrat", "`Sum ratio'"),
        ("distnr", "Pixel distance to nearest source"),
        ("high_noise", "High white noise fitted by G.P."),
    ]
    + [
        ("color_grad", "Rate of colour change"),
        ("fade", "Fade from G.P."),
        ("length_scale", "Length scale from G.P."),
        ("post_inflection", "Number of post-peak inflections"),
        ("score", "Score from G.P"),
    ]
)


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
