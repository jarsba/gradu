import sys
sys.path.append(snakemake.config['workdir'])

import os
import pickle

import numpy as np
import pandas as pd

from src.utils.synthetic_data_object import SynthDataObject
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET, COLUMNS_FOR_DATASET
from src.utils.path_utils import RESULTS_FOLDER
from base_lr import run_logistic_regression_on_3d, run_logistic_regression_on_2d

dataset_paths = snakemake.input

results = pd.DataFrame(
    columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "accuracy",
             "balanced_accuracy", "F1",
             "coefficients", "point_estimates", "variance_estimates", "confidence_intervals"])

for path in dataset_paths:
    print(path)
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

    dataset_columns = COLUMNS_FOR_DATASET[dataset_name]
    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in dataset_columns if col != target_column]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    for i in range(datasets):
        train_df = pd.DataFrame(dataset_tensor[i], columns=[COLUMNS_FOR_DATASET[dataset_name]])
        df_np = train_df.to_numpy()
        X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]

        accuracy_score, balanced_accuracy_score, f1_score, \
        coefficients, point_estimates, variance_estimates, confidence_intervals = run_logistic_regression_on_2d(df_np,
                                                                                                          X_train,
                                                                                                          y_train,
                                                                                                          X_test,
                                                                                                          y_test)

        results.append([experiment_id, dataset_name, i, query, epsilon, MCMC_algorithm, accuracy_score,
                        balanced_accuracy_score, f1_score, coefficients, point_estimates, variance_estimates,
                        confidence_intervals])

    dataset_tensor_stacked = dataset_tensor.reshape((n_datasets * n_rows, n_cols))
    train_df = pd.DataFrame(dataset_tensor_stacked, columns=[COLUMNS_FOR_DATASET[dataset_name]])
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    accuracy_score, balanced_accuracy_score, f1_score, \
    coefficients, point_estimates, variance_estimates, confidence_intervals = run_logistic_regression_on_3d(dataset_tensor,
                                                                                                      X_train,
                                                                                                      y_train, X_test,
                                                                                                      y_test)

    results.append([experiment_id, dataset_name, np.nan, query, epsilon, MCMC_algorithm, accuracy_score,
                    balanced_accuracy_score, f1_score, coefficients, point_estimates, variance_estimates,
                    confidence_intervals])

results.to_csv(snakemake.output[0], index=False)
