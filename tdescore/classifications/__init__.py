"""
Module for classification of ZTF sources
"""
from tdescore.classifications.crossmatch import classified
from tdescore.classifications.tde import all_tdes

full_source_list = list(set(all_tdes + classified["ztf_name"].tolist()))
