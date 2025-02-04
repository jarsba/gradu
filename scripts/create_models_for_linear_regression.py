import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
from typing import Literal
import pickle

sys.path.append(snakemake.config['workdir'])

import jax
from jax.config import config

config.update("jax_enable_x64", True)
print(f"Jax device count: {jax.local_device_count()}")

import torch

torch.set_default_dtype(torch.float64)

from arviz.data.inference_data import InferenceDataT

from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.string_utils import epsilon_float_to_str
from src.utils.timer import Timer
from src.utils.data_utils import transform_for_modeling
from src.utils.job_parameters import JobParameters
from src.utils.seed_utils import set_seed, get_seed
from src.utils.data_generator import create_dummy_dataset

"""
Create model with increasing number of variables in dataset and profiling enabled. Try to estimate the exponent for the runtime
using the number of canonical queries and the tree width.
"""

if __name__ == '__main__':

    parameter_combination_pickle_path = snakemake.input[0]
    parameter_combination_pickle_file = open(parameter_combination_pickle_path, "rb")
    parameter_combinations: JobParameters = pickle.load(parameter_combination_pickle_file)
    parameter_combination_pickle_file.close()

    epsilon = parameter_combinations.epsilon
    epsilon_str = epsilon_float_to_str(epsilon)
    experiment_id = parameter_combinations.experiment_id
    experiment_id_ctx.set(experiment_id)
    dataset_name = parameter_combinations.dataset
    query_list = parameter_combinations.query_list
    query_str = parameter_combinations.query_string
    laplace_approximation = parameter_combinations.laplace_approximation
    laplace_approximation_algorithm = parameter_combinations.laplace_approximation_algorithm
    algo = parameter_combinations.algo
    repeat_index = parameter_combinations.repeat_index

    seed = snakemake.config['seed']
    unique_seed = get_seed(seed, repeat_index)
    rng = set_seed(unique_seed)

    target_file = str(snakemake.output[0])

    storage_file_path = f"logs/napsu_linear_regression_test_storage_{experiment_id}.pickle"
    mode: Literal["append"] = "append"
    timer_file_path = "logs/napsu_linear_regression_test_timer.csv"

    storage = ExperimentStorage(file_path=storage_file_path, mode=mode)
    timer = Timer(file_path=timer_file_path, mode=mode)

    n_categories = int(dataset_name.split("x")[1])

    dataframe = create_dummy_dataset(n_columns=5, n_rows=10000, n_categories=n_categories)
    dataframe = transform_for_modeling(dataset_name, dataframe)

    n, d = dataframe.shape
    query = []
    delta = (n ** (-2))

    timer_meta = {
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "query": query,
        "epsilon": epsilon,
        "delta": delta,
        "MCMC_algo": algo,
        "laplace_approximation": laplace_approximation,
        "laplace_approximation_algorithm": laplace_approximation_algorithm,
        "repeat_index": repeat_index,
        "original_seed": unique_seed
    }

    pid = timer.start(f"Main run", **timer_meta)

    print(
        f"PARAMS: \n\tdataset name {dataset_name}\n\tcliques {query_str}\n\tMCMC algo {algo}\n\tepsilon {epsilon_str}\n\tdelta: {delta}\n\tLaplace approximation {laplace_approximation}")

    print("Initializing NapsuMQModel")
    model = NapsuMQModel()

    result: NapsuMQResult
    inf_data: InferenceDataT

    result, inf_data = model.fit(
        data=dataframe,
        dataset_name=dataset_name,
        rng=rng,
        epsilon=epsilon,
        delta=delta,
        column_feature_set=query,
        MCMC_algo=algo,
        use_laplace_approximation=laplace_approximation,
        return_inference_data=True,
        enable_profiling=False,
        laplace_approximation_algorithm=laplace_approximation_algorithm,
        laplace_approximation_forward_mode=True
    )

    timer.stop(pid)

    print("Writing model to file")
    result.store(target_file)

    inf_data.to_netcdf(f"logs/inf_data_linear_regression_{dataset_name}_{epsilon_str}e_{repeat_index}_repeat.nc")

    # Save storage and timer results every iteration
    storage.save_as_pickle(file_path=storage_file_path, experiment_id=experiment_id)
    timer.save(file_path=timer_file_path, mode=mode, index=False)
