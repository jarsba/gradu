import sys
sys.path.append(snakemake.config['workdir'])

from typing import Protocol

import pandas as pd
import numpy as np
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from base_clf import run_classification
from src.utils.data_utils import transform_for_classification


dataset_paths = snakemake.input
dataset_map = snakemake.config['original_datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> 'ScikitModel': ...

    def predict(self, X) -> np.ndarray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params) -> 'ScikitModel': ...


results = pd.DataFrame(columns=["dataset_name", "model_name", "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    print(f"Running classification on {path} dataset")
    train_df = pd.read_csv(path)

    dataset_name = inverted_dataset_map[path]
    train_df_transformed = transform_for_classification(dataset_name, train_df)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    test_df_transformed = transform_for_classification(dataset_name, test_df)

    # Check that both have equal columns
    assert set(list(train_df_transformed.columns.values)).symmetric_difference(
        set(list(test_df_transformed.columns.values))) == set()

    scores = run_classification(train_df_transformed, test_df_transformed, target_column)

    for model_score in scores:
        model_name, accuracy, balanced_accuracy, f1 = model_score
        results.loc[len(results)] = [dataset_name, model_name, accuracy, balanced_accuracy, f1]

results.to_csv(snakemake.output[0], index=False)
