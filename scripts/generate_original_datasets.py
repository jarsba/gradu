import sys

sys.path.append(snakemake.config['workdir'])

from src.utils.preprocess_dataset import \
    get_adult_train_no_discretization, get_adult_test_no_discretization, \
    get_adult_train_low_discretization, get_adult_test_low_discretization, \
    get_adult_train_high_discretization, get_adult_test_high_discretization, \
    get_adult_train_small, get_adult_test_small, \
    get_adult_train_large, get_adult_test_large, get_adult_train_independence_pruning, \
    get_adult_test_independence_pruning
from src.utils.path_utils import DATASETS_FOLDER

datasets = snakemake.config['datasets']
dataset_names = [key for key in datasets.keys()]
dataset_files = [value for value in datasets.values()]

PREPROCESSING_FUNCTIONS = {
    "adult_small": get_adult_train_small,
    "adult_large": get_adult_train_large,
    "adult_no_discretization": get_adult_train_no_discretization,
    "adult_low_discretization": get_adult_train_low_discretization,
    "adult_high_discretization": get_adult_train_high_discretization,
    "adult_small_test": get_adult_test_small,
    "adult_large_test": get_adult_test_large,
    "adult_no_discretization_test": get_adult_test_no_discretization,
    "adult_low_discretization_test": get_adult_test_low_discretization,
    "adult_high_discretization_test": get_adult_test_high_discretization,
    "adult_independence_pruning": get_adult_train_independence_pruning,
    "adult_independence_pruning_test": get_adult_test_independence_pruning
}

for dataset, target_file in zip(dataset_names, dataset_files):

    # Skip if the dataset is not in the preprocessing functions like for "adult", "binary3d" and "binary4d"
    if dataset not in PREPROCESSING_FUNCTIONS:
        print(f"Warning: Dataset not found in PREPROCESSING_FUNCTIONS. Skipping dataset: {dataset}")
        continue

    df = PREPROCESSING_FUNCTIONS[dataset](dataset_folder=DATASETS_FOLDER)
    df.to_csv(target_file, index=False)
