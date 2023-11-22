"""
Module for running sncosmo on ZTF data for nuclear sample
"""
import logging

from tdescore.sncosmo.run_sncosmo import batch_sncosmo

logging.getLogger().setLevel(logging.INFO)

batch_sncosmo()
