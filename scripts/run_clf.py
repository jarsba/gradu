from typing import List, Protocol

import pandas as pd
import numpy as np
import os
from .constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.path_utils import get_filename, get_dataset_name, get_metadata_from_synthetic_path, RESULTS_FOLDER

dataset_paths = snakemake.input[0]

class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> 'ScikitModel': ...

    def predict(self, X) -> np.ndarray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params) -> 'ScikitModel': ...


models = [
    DummyClassifier(strategy="most_frequent"),
    GradientBoostingClassifier(),
    LGBMClassifier(),
    XGBClassifier(),
    RandomForestClassifier(),
    LinearSVC(),
    MLPClassifier(hidden_layer_sizes=(100), solver='sgd', alpha=0.5)
]


def run_classification_on_adult(train_df, test_df, target_column, models: List[ScikitModel]):
    X_train, y_train = train_df.drop(columns=[target_column], axis=1), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column], axis=1), test_df[target_column]

    scores = []

    for model in models:
        model.fit(X_train, y_train)

        accuracy_score = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=-1, error_score='raise')
        balanced_accuracy_score = cross_val_score(model, X_test, y_test, scoring='balanced_accuracy', n_jobs=-1,
                                                  error_score='raise')

        f1_score = cross_val_score(model, X_test, y_test, scoring='f1', n_jobs=-1, error_score='raise')

        scores.append((type(model).__name__, accuracy_score, balanced_accuracy_score, f1_score))

        print(
            f'Model: {type(model).__name__} \t Accuracy: {np.mean(accuracy_score):.3f} ({np.std(accuracy_score):.3f}), Balanced accuracy: {np.mean(balanced_accuracy_score):.3f} ({np.std(balanced_accuracy_score):.3f}), F1: {np.mean(f1_score):.3f} ({np.std(f1_score):.3f})')

    return scores


synthetic_task = "synthetic_dataset" in get_filename(dataset_paths[0])

if synthetic_task:
    results = pd.DataFrame(
        columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "model_name", "accuracy", "balanced_accuracy", "F1"])
else:
    results = pd.DataFrame(columns=["dataset_name", "model_name", "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    train_df = pd.read_csv(path)

    dataset_name = get_dataset_name(path)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)

    scores = run_classification_on_adult(train_df, test_df, target_column, models)

    if synthetic_task:
        experiment_id, dataset_index, _, query, epsilon, MCMC_algorithm = get_metadata_from_synthetic_path(path)

        for model_score in scores:
            model_name, accuracy, balanced_accuracy, f1 = model_score
            results.append([experiment_id, dataset_name, dataset_index, query, epsilon, MCMC_algorithm, model_name, accuracy, balanced_accuracy, f1])
    else:
        for model_score in scores:
            model_name, accuracy, balanced_accuracy, f1 = model_score
            results.append([dataset_name, model_name, accuracy, balanced_accuracy, f1])

if synthetic_task:
    result_path = os.path.join(RESULTS_FOLDER, "synthetic_classification_results.csv")
    results.to_csv(result_path, index=False)
else:
    result_path = os.path.join(RESULTS_FOLDER, "original_classification_results.csv")
    results.to_csv(result_path, index=False)
