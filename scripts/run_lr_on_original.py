import sys
sys.path.append(snakemake.config['workdir'])

import pandas as pd
from base_lr import run_logistic_regression_on_2d
from constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from src.utils.data_utils import transform_for_classification

dataset_paths = snakemake.input
dataset_map = snakemake.config['original_datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}

results = pd.DataFrame(
    columns=["dataset_name", "accuracy", "balanced_accuracy", "F1", "coefficients", "point_estimates",
             "variance_estimates", "confidence_intervals"])

print(dataset_paths)

for path in dataset_paths:
    print(path)
    df = pd.read_csv(path)

    dataset_name = inverted_dataset_map[path]

    df_transformed = transform_for_classification(dataset_name, df)

    df_np = df_transformed.to_numpy()

    target_column: str = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in df_transformed.columns if col != target_column]

    X_train, y_train = df_transformed.drop(columns=[target_column]), df_transformed[target_column]

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    test_df_transformed = transform_for_classification(dataset_name, test_df)

    target_column_index = df_transformed.columns.get_loc(target_column)

    X_test, y_test = test_df_transformed.drop(columns=[target_column]), test_df_transformed[target_column]

    accuracy_score, balanced_accuracy_score, f1_score, \
    coefficients, point_estimates, variance_estimates, confidence_intervals = \
        run_logistic_regression_on_2d(df_np,
                                      X_train,
                                      y_train,
                                      X_test,
                                      y_test,
                                      return_confidence_intervals=True,
                                      col_to_predict=target_column_index)

    results.loc[len(results)] = [dataset_name, accuracy_score, balanced_accuracy_score, f1_score, coefficients,
                                 point_estimates,
                                 variance_estimates, confidence_intervals]

results.to_csv(snakemake.output[0], index=False)
