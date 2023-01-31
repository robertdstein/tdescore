"""
Module to analyse a lightcurve and extract metaparameters for further analysis
"""
# import pickle
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from astropy.stats import bayesian_blocks
# from nuztf.plot import alert_to_pandas
# from scipy.optimize import curve_fit
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.gaussian_process.kernels import ConstantKernel as C
# from sklearn.gaussian_process.kernels import (
#     DotProduct,
#     ExpSineSquared,
#     Matern,
#     RationalQuadratic,
#     WhiteKernel,
# )
# from tqdm import tqdm
#
# from tdescore.data import classified, ztf1_tde_list
# from tdescore.paths import ampel_cache_dir, lightcurve_dir

TIME_KEY = "mjd"
Y_KEY = "magpsf"
YERR_KEY = "sigmapsf"
