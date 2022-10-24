from typing import Protocol

import pandas as pd
import numpy as np
import os
from .constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from base_clf import run_classification_on_adult

from src.utils.path_utils import get_dataset_name, RESULTS_FOLDER

dataset_paths = snakemake.input[0]


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> 'ScikitModel': ...

    def predict(self, X) -> np.ndarray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params) -> 'ScikitModel': ...


results = pd.DataFrame(columns=["dataset_name", "model_name", "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    train_df = pd.read_csv(path)

    dataset_name = get_dataset_name(path)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)

    scores = run_classification_on_adult(train_df, test_df, target_column)

    for model_score in scores:
        model_name, accuracy, balanced_accuracy, f1 = model_score
        results.append([dataset_name, model_name, accuracy, balanced_accuracy, f1])

result_path = os.path.join(RESULTS_FOLDER, "original_classification_results.csv")
results.to_csv(result_path, index=False)
