from typing import Mapping

import numpy as np
import pandas as pd

from constants import TRAIN_DATASET_SIZE_MAP, TARGET_COLUMNS_FOR_DATASET
from data_utils import dataframe_list_to_tensor
from napsu_mq import NapsuMQResult
from rubins_rules import conf_int
from src.utils.confidence_interval_object import ConfidenceIntervalObject
from src.napsu_mq.logistic_regression import logistic_regression, logistic_regression_on_2d
import jax


def calculate_ci_coverage_objects(model: NapsuMQResult, test_dataset: np.ndarray, meta: Mapping,
                                  confidence_intervals: np.ndarray = np.linspace(0.05, 0.95, 19), n_repeats: int = 50,
                                  n_datasets: int = 100,
                                  rng: jax.random.PRNGKey = None,
                                  target_column_index: int = None) -> pd.DataFrame:
    if rng is None:
        rng = jax.random.PRNGKey(2534753284572)

    point_estimates, variance_estimates = logistic_regression_on_2d(test_dataset, return_intervals=False,
                                                                    return_results=False,
                                                                    col_to_predict=target_column_index)
    true_parameter_values = point_estimates
    sampling_rngs = jax.random.split(rng, n_repeats)
    dataset_name = model.meta['dataset_name']
    n_original_datapoints = TRAIN_DATASET_SIZE_MAP[dataset_name]

    ci_data_objects = []

    for i in range(n_repeats):
        for interval in confidence_intervals:
            datasets = model.generate(sampling_rngs[i], n_original_datapoints, n_datasets)

            datasets_np = dataframe_list_to_tensor(datasets)

            q, u = logistic_regression(datasets_np, add_constant=False, col_to_predict=target_column_index)

            for d in range(len(true_parameter_values)):
                q_i = q[:, d]
                u_i = u[:, d]

                ci_result = conf_int(q_i, u_i, interval)

                if np.isnan(ci_result[0]) or np.isnan(ci_result[1]):
                    print(f"WARNING: Confidence interval had nan: {ci_result}")

                true_param_value = true_parameter_values[d]

                print(
                    f"True param value: {true_param_value}, confidence interval: {ci_result[0]} - {ci_result[1]}")

                meta_dict = {key: model.meta[key] for key in meta}

                conf_int_object = ConfidenceIntervalObject(
                    original_dataset_name=dataset_name,
                    index=i,
                    n_datasets=n_datasets,
                    conf_int_range=interval,
                    conf_int_start=ci_result[0],
                    conf_int_end=ci_result[1],
                    conf_int_width=ci_result[1] - ci_result[0],
                    true_parameter_value=true_param_value,
                    contains_true_parameter=ci_result[0] <= true_param_value <= ci_result[1],
                    meta=meta_dict
                )

                ci_data_objects.append(conf_int_object)

    ci_df = pd.from_records([obj.to_dict() for obj in ci_data_objects])

    return ci_df
