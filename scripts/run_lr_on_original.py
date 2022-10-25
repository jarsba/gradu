import sys
sys.path.append(snakemake.config['workdir'])

import os
import pandas as pd
from base_lr import run_logistic_regression
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from src.utils.path_utils import get_dataset_name, get_filename, RESULTS_FOLDER
from src.utils.preprocess_dataset import clean_dataset, convert_to_int_array

dataset_paths = snakemake.input[0]

synthetic_task = "synthetic_dataset" in get_filename(dataset_paths[0])

results = pd.DataFrame(
    columns=["dataset_name", "accuracy", "balanced_accuracy", "F1", "coefficients", "point_estimates",
             "variance_estimates", "confidence_intervals"])

df_np_arrays = []

for path in dataset_paths:
    df = pd.read_csv(path)

    dataset_name = get_dataset_name(path)

    df = clean_dataset(df, dataset_name)

    df_np = convert_to_int_array(df)
    df_np_arrays.append(df_np)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in df.columns if col != target_column]

    X_train, y_train = df.drop(columns=[target_column]), df[target_column]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    accuracy_score, balanced_accuracy_score, f1_score, \
    coefficients, point_estimates, variance_estimates, confidence_intervals = run_logistic_regression(df_np, X_train,
                                                                                                      y_train, X_test,
                                                                                                      y_test)

    results.append([dataset_name, accuracy_score, balanced_accuracy_score, f1_score, coefficients, point_estimates,
                    variance_estimates, confidence_intervals])

result_path = os.path.join(RESULTS_FOLDER, "original_logistic_regression_results.csv")
results.to_csv(result_path, index=False)
