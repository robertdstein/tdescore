"""
Module for downloading, analysing and parsing data used for training classifier
"""
import logging

import pandas as pd

from tdescore.metadata.parse import parse_metadata
from tdescore.paths import combined_metadata_path

logger = logging.getLogger(__name__)


if not combined_metadata_path.exists():
    parse_metadata()

combined_metadata = pd.read_json(combined_metadata_path)
