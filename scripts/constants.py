TARGET_COLUMNS_FOR_DATASET = {
    "adult": "compensation",
    "binary4d": "D",
    "binary3d": "C"
}

TRAIN_DATASET_FOR_DATASET = {
    "adult": "data/datasets/cleaned_adult_train_data.csv",
    "binary4d": "data/datasets/binary4d.csv",
    "binary3d": "data/datasets/binary3d.csv"
}

TEST_DATASETS_FOR_DATASET = {
    "adult": "data/datasets/cleaned_adult_test_data.csv",
    "binary4d": "data/datasets/binary4d_test.csv",
    "binary3d": "data/datasets/binary3d_test.csv"
}

TRAIN_DATASET_SIZE_MAP = {
    "adult": 30162,
    "binary4d": 100000,
    "binary3d": 100000,
}

TEST_DATASET_SIZE_MAP = {
    "adult": 15060,
    "binary4d": 50000,
    "binary3d": 50000,
}