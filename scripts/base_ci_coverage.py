import numpy as np
import pandas as pd
import d3p
from src.napsu_mq.napsu_mq import NapsuMQResult
from src.napsu_mq.rubins_rules import conf_int, non_negative_conf_int
from src.utils.confidence_interval_object import ConfidenceIntervalObject
from src.utils.data_utils import transform_for_ci_coverage, dataframe_list_to_tensor
from src.napsu_mq.logistic_regression import logistic_regression, logistic_regression_on_2d
import jax


def calculate_ci_coverage_objects(model: NapsuMQResult, test_dataset: np.ndarray, meta: dict,
                                  confidence_intervals: np.ndarray = np.linspace(0.05, 0.95, 19), n_repeats: int = 50,
                                  n_datasets: int = 100, n_syn_datapoints: int = None,
                                  rng: jax.random.PRNGKey = None,
                                  target_column_index: int = None) -> pd.DataFrame:
    if rng is None:
        rng = jax.random.PRNGKey(2534753284572)

    if n_syn_datapoints is None:
        n_syn_datapoints = test_dataset.shape[0]

    point_estimates, variance_estimates = logistic_regression_on_2d(test_dataset, col_to_predict=target_column_index,
                                                                    return_intervals=False, return_results=False,
                                                                    add_constant=False)
    true_parameter_values = point_estimates.flatten()

    sampling_rngs = d3p.random.split(rng, n_repeats * len(confidence_intervals))
    #dataset_name = model.meta['dataset_name']
    #n_original_datapoints = TRAIN_DATASET_SIZE_MAP[dataset_name]
    dataset_name = "binary3d"
    n_original_datapoints = 100000
    ci_data_objects = []

    for i in range(n_repeats):
        for j, interval in enumerate(confidence_intervals):
            print(f"Running CI coverage for dataset {dataset_name}, index {i} and interval {interval}", flush=True)

            rng_index = i * len(confidence_intervals) + j
            rng = sampling_rngs[rng_index]

            datasets = model.generate(rng, n_syn_datapoints, n_datasets, single_dataframe=False)

            datasets_transformed = [transform_for_ci_coverage(dataset_name, dataset) for dataset in datasets]

            datasets_np = dataframe_list_to_tensor(datasets_transformed)

            q, u = logistic_regression(datasets_np, col_to_predict=target_column_index, return_intervals=False,
                                       return_results=False, add_constant=False)

            for d in range(len(true_parameter_values)):
                q_i = q[:, d]
                u_i = u[:, d]

                # Add check for huge confidence intervals and remove them from the logistic regression point and variance
                # estimates
                inds = (u_i < 1000)
                q_i = q_i[inds]
                u_i = u_i[inds]

                if len(u_i) == 0:
                    q_i = np.array(np.nan)
                    u_i = np.array(np.nan)

                ci_result = conf_int(q_i, u_i, interval)

                if np.isnan(ci_result[0]) or np.isnan(ci_result[1]):
                    print(f"WARNING: Confidence interval had nan: {ci_result}")

                ci_result_nn = non_negative_conf_int(
                    q_i, u_i, interval, n_syn_datapoints, n_original_datapoints
                )

                if np.isnan(ci_result_nn[0]) or np.isnan(ci_result_nn[1]):
                    print(f"WARNING: Confidence interval had nan: {ci_result_nn}")

                true_param_value = true_parameter_values[d]

                print(
                    f"True param value: {true_param_value}, confidence interval: {ci_result[0]} - {ci_result[1]}")

                print(
                    f"True param value: {true_param_value}, non-negative confidence interval: {ci_result_nn[0]} - {ci_result_nn[1]}")

                contains_true_value = ci_result[0] <= true_param_value <= ci_result[1]
                contains_true_value_nn = ci_result_nn[0] <= true_param_value <= ci_result_nn[1]

                conf_int_object = ConfidenceIntervalObject(
                    original_dataset_name=dataset_name,
                    index=i,
                    n_datasets=n_datasets,
                    conf_int_range=interval,
                    conf_int_start=ci_result[0],
                    conf_int_end=ci_result[1],
                    conf_int_width=ci_result[1] - ci_result[0],
                    true_parameter_value=true_param_value,
                    contains_true_parameter=contains_true_value,
                    nn_conf_int_start=ci_result_nn[0],
                    nn_conf_int_end=ci_result_nn[1],
                    nn_conf_int_width=ci_result_nn[1] - ci_result_nn[0],
                    contains_true_parameter_nn=contains_true_value_nn,
                    parameter_index=d+1,
                    meta=meta
                )

                ci_data_objects.append(conf_int_object)

    ci_df = pd.DataFrame.from_records([obj.to_dict() for obj in ci_data_objects])

    return ci_df
