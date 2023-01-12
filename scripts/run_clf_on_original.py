import sys

sys.path.append(snakemake.config['workdir'])

from typing import Protocol

import pandas as pd
import numpy as np
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from base_clf import run_classification

dataset_paths = snakemake.input
dataset_map = snakemake.config['datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> 'ScikitModel': ...

    def predict(self, X) -> np.ndarray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params) -> 'ScikitModel': ...


results = pd.DataFrame(columns=["dataset_name", "model_name", "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    train_df = pd.read_csv(path)

    dataset_name = inverted_dataset_map[path]

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    scores = run_classification(train_df, test_df, target_column)

    for model_score in scores:
        model_name, accuracy, balanced_accuracy, f1 = model_score
        results.loc[len(results)] = [dataset_name, model_name, accuracy, balanced_accuracy, f1]

results.to_csv(snakemake.output[0], index=False)
