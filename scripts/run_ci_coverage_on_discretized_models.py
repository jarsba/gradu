import sys

from constants import TEST_DATASETS_FOR_DATASET, TARGET_COLUMNS_FOR_DATASET
from napsu_mq import NapsuMQResult
import pandas as pd
sys.path.append(snakemake.config['workdir'])

from scripts.base_ci_coverage import calculate_ci_coverage_objects

models = snakemake.input

for model_path in models:
    print(f"Generating data for model {model_path}")
    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    discretization_level = meta_info['discretization']
    experiment_id = meta_info['experiment_id']

    test_df_path = TEST_DATASETS_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    target_column: str = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    target_column_index = test_df.columns.get_loc(target_column)

    ci_coverage_results = calculate_ci_coverage_objects(model, , target_column_index=target_column_index)