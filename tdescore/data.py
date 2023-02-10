"""
Module for downloading, analysing and parsing data used for training classifier
"""
import logging

from tdescore.metadata.parse import parse_metadata
from tdescore.paths import combined_metadata_path

logger = logging.getLogger(__name__)


if not combined_metadata_path.exists():
    parse_metadata()
