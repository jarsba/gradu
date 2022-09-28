import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils.path_utils import get_dataset_name, get_filename, get_metadata_from_filename

dataset_paths = snakemake.input[0]
print(dataset_paths)

TARGET_COLUMNS_FOR_DATASET = {
    "adult": "compensation",
    "binary4d": "D"
}

TEST_DATASETS_FOR_DATASET = {
    "adult": "data/datasets/cleaned_adult_data_v2_test.csv",
    "binary4d": "data/datasets/binary4d_test.csv"
}

synthetic_task = "synthetic_dataset" in get_filename(dataset_paths[0])

if synthetic_task:
    results = pd.DataFrame(columns=["dataset_name", "index", "epsilon", "MCMC_algorithm", "accuracy", "coefficients"])
else:
    results = pd.DataFrame(columns=["dataset_name", "accuracy", "coefficients"])

for path in dataset_paths:
    df = pd.read_csv(path)

    dataset_name = get_dataset_name(path)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in df.columns if col != target_column]

    X, y = df.drop(columns=[target_column]), df[target_column]

    clf = LogisticRegression(random_state=0).fit(X, y)
    coeficcients = clf.coef_

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)
    test_X, test_y = test_df.drop(columns=[target_column]), test_df[target_column]

    accuracy = clf.score(test_X, test_y)

    if synthetic_task:
        index, _, epsilon, MCMC_algorithm = get_metadata_from_filename(path)
        results.append([dataset_name, index, epsilon, MCMC_algorithm, accuracy, coeficcients])
    else:
        results.append([dataset_name, accuracy, coeficcients])

if synthetic_task:
    result_path = os.path.join("results", "synthetic_logistic_regression_results.csv")
    results.to_csv(result_path)
else:
    result_path = os.path.join("results", "original_logistic_regression_results.csv")
    results.to_csv(result_path)
