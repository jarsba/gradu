import sys

sys.path.append(snakemake.config['workdir'])

import pickle

import numpy as np
import pandas as pd

from src.utils.synthetic_data_object import SynthDataObject
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET, COLUMNS_FOR_DATASET
from src.utils.data_utils import transform_for_classification
from base_lr import run_logistic_regression_on_3d, run_logistic_regression_on_2d
from src.utils.classification_utils import compare_and_fill_missing_columns
from src.utils.seed_utils import set_seed

seed = snakemake.config['seed']
rng = set_seed(seed)

dataset_paths = snakemake.input

results = pd.DataFrame(
    columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "accuracy",
             "balanced_accuracy", "F1",
             "coefficients", "point_estimates", "variance_estimates"])

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

    dataset_columns = COLUMNS_FOR_DATASET[dataset_name]
    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in dataset_columns if col != target_column]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    test_df_transformed = transform_for_classification(dataset_name, test_df)

    target_column_index = test_df_transformed.columns.get_loc(target_column)

    X_test, y_test = test_df_transformed.drop(columns=[target_column]), test_df_transformed[target_column]

    for i in range(datasets):
        print(f"Running logistic regression on {path} and index {i}")
        train_df = pd.DataFrame(dataset_tensor[i], columns=COLUMNS_FOR_DATASET[dataset_name])
        train_df_transformed = transform_for_classification(dataset_name, train_df)
        train_df_transformed = compare_and_fill_missing_columns(test_df_transformed, train_df_transformed)

        # Check that both have equal columns
        assert set(list(train_df_transformed.columns.values)).symmetric_difference(
            set(list(test_df_transformed.columns.values))) == set()

        df_np = train_df_transformed.to_numpy()
        X_train, y_train = train_df_transformed.drop(columns=[target_column]), train_df_transformed[target_column]

        accuracy_score, balanced_accuracy_score, f1_score, \
        coefficients, point_estimates, variance_estimates = run_logistic_regression_on_2d(df_np,
                                                                                          X_train,
                                                                                          y_train,
                                                                                          X_test,
                                                                                          y_test,
                                                                                          col_to_predict=target_column_index)

        results.loc[len(results)] = [experiment_id, dataset_name, i, query, epsilon, MCMC_algorithm, accuracy_score,
                                     balanced_accuracy_score, f1_score, coefficients, point_estimates,
                                     variance_estimates]

    print(f"Running logistic regression on whole dataset: {path}")
    # Classify the whole synthetic dataset
    dataset_tensor_stacked = dataset_tensor.reshape((n_datasets * n_rows, n_cols))
    train_df = pd.DataFrame(dataset_tensor_stacked, columns=COLUMNS_FOR_DATASET[dataset_name])
    train_df_transformed = transform_for_classification(dataset_name, train_df)
    train_df_transformed = compare_and_fill_missing_columns(test_df_transformed, train_df_transformed)

    # Check that both have equal columns
    assert set(list(train_df_transformed.columns.values)).symmetric_difference(
        set(list(test_df_transformed.columns.values))) == set()

    df_np = train_df_transformed.to_numpy()

    X_train, y_train = train_df_transformed.drop(columns=[target_column]), train_df_transformed[target_column]
    accuracy_score, balanced_accuracy_score, f1_score, \
    coefficients, point_estimates, variance_estimates = run_logistic_regression_on_2d(
        df_np, X_train, y_train, X_test, y_test, col_to_predict=target_column_index)

    results.loc[len(results)] = [experiment_id, dataset_name, np.nan, query, epsilon, MCMC_algorithm,
                                 accuracy_score,
                                 balanced_accuracy_score, f1_score, coefficients, point_estimates,
                                 variance_estimates]

    results.to_csv(snakemake.output[0], index=False)
