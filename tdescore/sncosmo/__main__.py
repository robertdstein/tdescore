"""
Module for running sncosmo on ZTF data for nuclear sample
"""
import logging

from tdescore.sncosmo.run_sncosmo import batch_sncosmo

logging.getLogger("tdescore").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

batch_sncosmo()
