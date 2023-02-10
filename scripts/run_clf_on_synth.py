import sys

sys.path.append(snakemake.config['workdir'])

import pandas as pd
import os
import numpy as np
from src.utils.synthetic_data_object import SynthDataObject
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET, COLUMNS_FOR_DATASET
from base_clf import run_classification
from src.utils.path_utils import RESULTS_FOLDER
from src.utils.data_utils import transform_for_classification
import pickle

dataset_paths = snakemake.input

results = pd.DataFrame(
    columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "model_name",
             "accuracy", "balanced_accuracy", "F1"])

for path in dataset_paths:
    synth_file = open(path, "rb")
    synth_data_object: SynthDataObject = pickle.load(synth_file)
    synth_file.close()
    dataset_tensor = synth_data_object.synth_data

    n_datasets, n_rows, n_cols = dataset_tensor.shape

    datasets = synth_data_object.n_datasets
    dataset_name = synth_data_object.original_dataset
    experiment_id = synth_data_object.experiment_id
    query = synth_data_object.queries
    epsilon = synth_data_object.epsilon
    MCMC_algorithm = synth_data_object.inference_algorithm

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    test_df_transformed = transform_for_classification(dataset_name, test_df)

    for i in range(datasets):

        print(f"Running classification on {path} and index {i}")

        train_df = pd.DataFrame(dataset_tensor[i], columns=COLUMNS_FOR_DATASET[dataset_name])

        train_df_transformed = transform_for_classification(dataset_name, train_df)


        scores = run_classification(train_df_transformed, test_df_transformed, target_column)

        for model_score in scores:
            model_name, accuracy, balanced_accuracy, f1 = model_score
            results.loc[len(results)] = [experiment_id, dataset_name, i, query, epsilon, MCMC_algorithm, model_name,
                                         accuracy,
                                         balanced_accuracy, f1]

    # Classify the whole synthetic dataset
    dataset_tensor_stacked = dataset_tensor.reshape((n_datasets * n_rows, n_cols))
    train_df = pd.DataFrame(dataset_tensor_stacked, columns=COLUMNS_FOR_DATASET[dataset_name])
    train_df_transformed = transform_for_classification(dataset_name, train_df)
    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    scores = run_classification(train_df_transformed, test_df_transformed, target_column)

    for model_score in scores:
        model_name, accuracy, balanced_accuracy, f1 = model_score
        results.loc[len(results)] = [experiment_id, dataset_name, np.nan, query, epsilon, MCMC_algorithm, model_name,
                                     accuracy,
                                     balanced_accuracy, f1]

results.to_csv(snakemake.output[0], index=False)
