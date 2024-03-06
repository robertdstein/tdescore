"""
Module to train the classifier
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier

from tdescore.classifier.balance import balance_train_data
from tdescore.classifier.collate import convert_to_train_dataset, get_classified_sources
from tdescore.classifier.features import default_columns

logger = logging.getLogger(__name__)


def train_classifier(
    n_iter: int = 10,
    train_sources: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    n_estimator_set: list[float] = None,
    balance_data: bool = True,
) -> tuple[pd.DataFrame, dict[float, XGBClassifier]]:
    """
    Function to train the classifier

    :param n_iter: Number of iterations to run
    :param train_sources: Sources to use. If None, use all classified sources
    :param columns: Columns to use
    :param n_estimator_set: Set of n_estimators to use
    :param balance_data: Whether to balance the data
    :return: Results of the training, Classifier
    """
    if train_sources is None:
        train_sources = get_classified_sources()

    if columns is None:
        columns = default_columns

    assert n_iter > 0, "n_iter must be positive"

    if n_estimator_set is None:
        n_estimator_set = [75.0, 100.0, 125.0]

    model_class = XGBClassifier
    logger.info(f"Model is {model_class.__name__}")

    all_all_res = []

    clfs = {}

    for n_estimators in n_estimator_set:
        kwargs = {
            "n_estimators": int(n_estimators),
            "eval_metric": "aucpr",
            "subsample": 0.7,
        }

        all_test_true = []
        all_test_pred = []
        all_test_pred_bool = []

        all_res = None

        for i in tqdm(range(n_iter)):
            # Randomly reorder all sources
            train_sources = train_sources.sample(frac=1).reset_index(drop=True)

            data_to_use = convert_to_train_dataset(train_sources, columns=columns)

            mask = train_sources["class"].to_numpy()

            nan_mask = np.array([np.sum(np.isnan(x)) > 0 for x in data_to_use])

            data_to_use = data_to_use[~nan_mask]
            mask = mask[~nan_mask]

            n_tde = np.sum(mask)

            # prepare cross validation
            kfold = StratifiedKFold(n_tde)

            all_pred_mask = np.ones_like(mask) * np.nan
            probs = np.ones_like(mask) * np.nan

            for train, test in kfold.split(data_to_use, mask):
                x_train, x_test = data_to_use[train], data_to_use[test]
                y_train, y_test = mask[train], mask[test]

                if balance_data:
                    x_train, y_train = balance_train_data(x_train, y_train)

                clf = model_class(**kwargs).fit(
                    x_train,
                    y_train,
                )

                probs[test] = clf.predict_proba(x_test).T[1]
                all_pred_mask[test] = clf.predict(x_test)

                pred_mask = clf.predict(x_test).astype(bool)

                all_test_true += list(y_test)
                all_test_pred += list(clf.predict_proba(x_test).T[1])
                all_test_pred_bool += list(pred_mask)

            if all_res is None:
                all_res = pd.DataFrame(
                    {
                        "ztf_name": train_sources[~nan_mask]["ztf_name"],
                        "class": mask,
                        f"probs_{i}": probs,
                    }
                ).set_index("ztf_name")
            else:
                new = pd.DataFrame(
                    {
                        "ztf_name": train_sources[~nan_mask]["ztf_name"],
                        f"probs_{i}": probs,
                    }
                )
                all_res = all_res.join(new.set_index("ztf_name"))

        score = accuracy_score(all_test_true, all_test_pred_bool)
        balanced_score = balanced_accuracy_score(all_test_true, all_test_pred_bool)

        frac = np.sum(np.array(all_test_true)[np.array(all_test_pred_bool)]) / np.sum(
            all_test_true
        )
        purity = np.sum(np.array(all_test_true)[np.array(all_test_pred_bool)]) / np.sum(
            all_test_pred_bool
        )

        roc_area = roc_auc_score(all_test_true, all_test_pred)

        precision, recall, _ = precision_recall_curve(all_test_true, all_test_pred)
        pr_area = auc(recall, precision)

        logger.info(f"N estimators = {n_estimators:.0f}")
        logger.info(f"Global score {100. * score:.2f}")
        logger.info(f"Global balanced score {100. * balanced_score:.2f}")
        logger.info(f"Global recovery fraction {100. * frac:.1f}%")
        logger.info(f"Global purity {100. * purity:.1f}%")
        logger.info(f"Global roc area {100. * roc_area:.2f}")
        logger.info(f"Global precision/recall area {100. * pr_area:.2f}")

        clfs[n_estimators] = clf

        all_all_res.append(
            {
                "n_estimator": n_estimators,
                "score": score,
                "balanced_score": balanced_score,
                "recall": frac,
                "precision": purity,
                "roc_area": roc_area,
                "precision_recall_area": pr_area,
                "all_res": all_res,
            }
        )

    all_all_res = pd.DataFrame(all_all_res)

    return all_all_res, clfs
