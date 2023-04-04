TARGET_COLUMNS_FOR_DATASET = {
    "adult": "compensation",
    "binary4d": "D",
    "binary3d": "C",
    "adult_small": "compensation",
    "adult_large": "compensation",
    "adult_no_discretization": "compensation",
    "adult_low_discretization": "compensation",
    "adult_high_discretization": "compensation",
    "adult_independence_pruning": "compensation",
}

TRAIN_DATASET_FOR_DATASET = {
    "adult": "data/datasets/cleaned_adult_train_data.csv",
    "binary4d": "data/datasets/binary4d.csv",
    "binary3d": "data/datasets/binary3d.csv",
    "adult_small": "data/datasets/adult_small.csv",
    "adult_large": "data/datasets/adult_large.csv",
    "adult_no_discretization": "data/datasets/adult_no_discretization.csv",
    "adult_low_discretization": "data/datasets/adult_low_discretization.csv",
    "adult_high_discretization": "data/datasets/adult_high_discretization.csv",
    "adult_independence_pruning": "data/datasets/adult_independence_pruning.csv"
}

TEST_DATASETS_FOR_DATASET = {
    "adult": "data/datasets/cleaned_adult_test_data.csv",
    "binary4d": "data/datasets/binary4d_test.csv",
    "binary3d": "data/datasets/binary3d_test.csv",
    "adult_small": "data/datasets/adult_small_test.csv",
    "adult_large": "data/datasets/adult_large_test.csv",
    "adult_no_discretization": "data/datasets/adult_no_discretization_test.csv",
    "adult_low_discretization": "data/datasets/adult_low_discretization_test.csv",
    "adult_high_discretization": "data/datasets/adult_high_discretization_test.csv",
    "adult_independence_pruning": "data/datasets/adult_independence_pruning_test.csv"
}

TRAIN_DATASET_SIZE_MAP = {
    "adult": 30162,
    "binary4d": 100000,
    "binary3d": 100000,
    "adult_small": 30162,
    "adult_large": 30162,
    "adult_no_discretization": 30162,
    "adult_low_discretization": 30162,
    "adult_high_discretization": 30162,
    "adult_independence_pruning": 30162,
}

TEST_DATASET_SIZE_MAP = {
    "adult": 15060,
    "binary4d": 50000,
    "binary3d": 50000,
    "adult_small": 15060,
    "adult_large": 15060,
    "adult_no_discretization": 15060,
    "adult_low_discretization": 15060,
    "adult_high_discretization": 15060,
    "adult_independence_pruning": 30162,

}

COLUMNS_FOR_DATASET = {
    "adult": ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'hours-per-week', 'native-country', 'compensation'],
    "binary4d": ['A', 'B', 'C', 'D'],
    "binary3d": ['A', 'B', 'C'],
    "adult_small": ["education-num", "marital-status", "sex", "age", "hours-per-week", "compensation"],
    "adult_large": ["age", "sex", "education-num", "hours-per-week", "workclass", "marital-status",
                    "had-capital-gains", "had-capital-losses", "compensation"],
    "adult_no_discretization": ['age', 'education-num', "sex", 'hours-per-week', 'workclass', 'marital-status',
                                'had-capital-gains', 'had-capital-losses', 'compensation'],
    "adult_low_discretization": ['age', 'education-num', "sex", 'hours-per-week', 'workclass', 'marital-status',
                                'had-capital-gains', 'had-capital-losses', 'compensation'],
    "adult_high_discretization": ['age', 'education-num', "sex", 'hours-per-week', 'workclass', 'marital-status',
                                'had-capital-gains', 'had-capital-losses', 'compensation'],
    "adult_independence_pruning": ["education-num", "sex", "age", "hours-per-week", "compensation"]
}


TRUE_COEFFICIENTS_FOR_DATASETS = {
    "binary3d": [1.0, 0.0],
    "binary4d": [1.0, 0.0, 2.0]
}
