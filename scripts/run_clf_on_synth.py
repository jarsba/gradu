import sys
sys.path.append(snakemake.config['workdir'])

import pandas as pd
import os
import numpy as np
from src.utils.synthetic_data_object import SynthDataObject
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET, COLUMNS_FOR_DATASET
from base_clf import run_classification_on_adult
from src.utils.path_utils import RESULTS_FOLDER
import pickle

dataset_paths = snakemake.input

print(dataset_paths)

results = pd.DataFrame(
    columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "model_name",
             "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    print(path)
    synth_file = open(path, "rb")
    synth_data_object: SynthDataObject = pickle.load(synth_file)
    dataset_tensor = synth_data_object.synth_data

    n_datasets, n_rows, n_cols = dataset_tensor.shape

    datasets = synth_data_object.n_datasets
    dataset_name = synth_data_object.original_dataset
    experiment_id = synth_data_object.experiment_id
    query = synth_data_object.query
    epsilon = synth_data_object.epsilon
    MCMC_algorithm = synth_data_object.inference_algorithm

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)

    for i in range(datasets):
        train_df = pd.DataFrame(dataset_tensor[i], columns=[COLUMNS_FOR_DATASET[dataset_name]])

        scores = run_classification_on_adult(train_df, test_df, target_column)

        for model_score in scores:
            model_name, accuracy, balanced_accuracy, f1 = model_score
            results.append(
                [experiment_id, dataset_name, i, query, epsilon, MCMC_algorithm, model_name, accuracy,
                 balanced_accuracy, f1])

    # Classify the whole synthetic dataset
    dataset_tensor_stacked = dataset_tensor.reshape((n_datasets * n_rows, n_cols))
    train_df = pd.DataFrame(dataset_tensor_stacked, columns=[COLUMNS_FOR_DATASET[dataset_name]])
    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)
    scores = run_classification_on_adult(train_df, test_df, target_column)

    for model_score in scores:
        model_name, accuracy, balanced_accuracy, f1 = model_score
        results.append(
            [experiment_id, dataset_name, np.nan, query, epsilon, MCMC_algorithm, model_name, accuracy,
             balanced_accuracy, f1])


result_path = os.path.join(RESULTS_FOLDER, "synthetic_classification_results.csv")
results.to_csv(result_path, index=False)
