import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from .constants import TARGET_COLUMNS_FOR_DATASET, TEST_DATASETS_FOR_DATASET
from src.napsu_mq.logistic_regression import logistic_regression
from src.utils.path_utils import get_dataset_name, get_filename, get_metadata_from_synthetic_path, RESULTS_FOLDER

dataset_paths = snakemake.input[0]
print(dataset_paths)

synthetic_task = "synthetic_dataset" in get_filename(dataset_paths[0])

if synthetic_task:
    results = pd.DataFrame(
        columns=["experiment_id", "dataset_name", "dataset_index", "query", "epsilon", "MCMC_algorithm", "accuracy", "balanced_accuracy", "F1",
                 "coefficients"])
else:
    results = pd.DataFrame(columns=["dataset_name", "accuracy", "balanced_accuracy", "F1", "coefficients"])


for path in dataset_paths:
    df = pd.read_csv(path)

    dataset_name = get_dataset_name(path)

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    feature_columns = [col for col in df.columns if col != target_column]

    X_train, y_train = df.drop(columns=[target_column]), df[target_column]

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    coeficcients = model.coef_

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(path)
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    accuracy_score = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=-1, error_score='raise')
    balanced_accuracy_score = cross_val_score(model, X_test, y_test, scoring='balanced_accuracy', n_jobs=-1,
                                              error_score='raise')
    f1_score = cross_val_score(model, X_test, y_test, scoring='f1', n_jobs=-1, error_score='raise')

    if synthetic_task:
        experiment_id, dataset_index, _, query, epsilon, MCMC_algorithm = get_metadata_from_synthetic_path(path)
        results.append([experiment_id, dataset_name, dataset_index, query, epsilon, MCMC_algorithm, accuracy_score, balanced_accuracy_score, f1_score, coeficcients])
    else:
        results.append([dataset_name, accuracy_score, balanced_accuracy_score, f1_score, coeficcients])

if synthetic_task:
    result_path = os.path.join(RESULTS_FOLDER, "synthetic_logistic_regression_results.csv")
    results.to_csv(result_path, index=False)
else:
    result_path = os.path.join(RESULTS_FOLDER, "original_logistic_regression_results.csv")
    results.to_csv(result_path, index=False)
