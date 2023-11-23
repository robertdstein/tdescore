"""
This module contains functions to balance the training data
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def balance_train_data(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use SMOTE to balance the training data

    :param x_train: Training data
    :param y_train: Training labels
    :return: Updated training data and labels
    """
    # Augment with SMOTE

    over = SMOTE(sampling_strategy=0.5)
    steps = [("o", over)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    x_train, y_train = pipeline.fit_resample(x_train, y_train)

    return x_train, y_train
