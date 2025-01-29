"""
Main module which analyses all lightcurves
"""
import logging

from tdescore.lightcurve.analyse import batch_analyse

logging.getLogger().setLevel(logging.INFO)

batch_analyse(overwrite=False)
