import sys
sys.path.append(snakemake.config['workdir'])

from constants import TARGET_COLUMNS_FOR_DATASET, TRAIN_DATASET_FOR_DATASET
from src.napsu_mq.napsu_mq import NapsuMQResult
import pandas as pd
from src.utils.data_utils import transform_for_ci_coverage
from scripts.base_ci_coverage import calculate_ci_coverage_objects

models = snakemake.input

for model_path in models:
    print(f"Generating data for model {model_path}")
    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    query_str = meta_info['query_str']
    experiment_id = meta_info['experiment_id']

    test_df_path = TRAIN_DATASET_FOR_DATASET[dataset_name]
    test_df = pd.read_csv(test_df_path)

    test_df_transformed = transform_for_ci_coverage("dataset_name", test_df)
    test_df_np = test_df_transformed.to_numpy()

    target_column = TARGET_COLUMNS_FOR_DATASET[dataset_name]
    target_column = test_df.columns.get_loc(target_column)

    meta = {
        'query_str': query_str,
        'experiment_id': experiment_id,
        'epsilon': epsilon,
    }

    ci_coverage_results = calculate_ci_coverage_objects(model, test_dataset=test_df_np, meta=meta, target_column_index=target_column)