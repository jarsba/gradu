import sys
from typing import Literal
import pickle
sys.path.append(snakemake.config['workdir'])

from jax.config import config
config.update("jax_enable_x64", True)

import torch
torch.set_default_dtype(torch.float64)

from src.utils.job_parameters import JobParameters

import pandas as pd
from arviz.data.inference_data import InferenceDataT

from src.utils.timer import Timer
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.string_utils import epsilon_float_to_str
from src.utils.data_utils import transform_for_modeling
from src.utils.seed_utils import set_seed

if __name__ == '__main__':

    seed = snakemake.config['seed']
    rng = set_seed(seed)

    parameter_combination_pickle_path = snakemake.input[0]
    parameter_combination_pickle_file = open(parameter_combination_pickle_path, "rb")
    parameter_combinations: JobParameters = pickle.load(parameter_combination_pickle_file)
    parameter_combination_pickle_file.close()

    epsilon = parameter_combinations.epsilon
    epsilon_str = epsilon_float_to_str(epsilon)
    experiment_id = parameter_combinations.experiment_id
    experiment_id_ctx.set(experiment_id)
    dataset_name = parameter_combinations.dataset
    dataset_path = parameter_combinations.dataset_path
    query_list = parameter_combinations.query_list
    query_str = parameter_combinations.query_string
    laplace_approximation = parameter_combinations.laplace_approximation
    laplace_approximation_algorithm = parameter_combinations.laplace_approximation_algorithm
    algo = parameter_combinations.algo

    print(epsilon)
    print(dataset_name)
    print(dataset_path)
    print(query_list)

    target_file = str(snakemake.output[0])


    storage_file_path = "napsu_experiment_storage_output.csv"
    mode: Literal["append"] = "append"
    timer_file_path = "napsu_MCMC_time_vs_epsilon_comparison.csv"

    storage = ExperimentStorage(file_path=storage_file_path, mode=mode)
    timer = Timer(file_path=timer_file_path, mode=mode)

    dataframe = pd.read_csv(dataset_path)

    dataframe = transform_for_modeling(dataset_name, dataframe)

    n, d = dataframe.shape
    delta = (n ** (-2))

    timer_meta = {
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "query": query_str,
        "epsilon": epsilon,
        "delta": delta,
        "MCMC_algo": algo,
        "laplace_approximation": laplace_approximation,
        "laplace_approximation_algorithm": laplace_approximation_algorithm
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
        column_feature_set=query_list,
        MCMC_algo=algo,
        use_laplace_approximation=laplace_approximation,
        return_inference_data=True,
        enable_profiling=True,
        laplace_approximation_algorithm=laplace_approximation_algorithm,
        laplace_approximation_forward_mode=True
    )

    timer.stop(pid)

    result.store(target_file)

    inf_data.to_netcdf(f"logs/inf_data_napsu_original_model_{dataset_name}_{epsilon}e_{query_str}.nc")

    # Save storage and timer results every iteration
    storage.save(file_path=storage_file_path, mode=mode)
    timer.save(file_path=timer_file_path, mode=mode)

    storage.save(file_path=storage_file_path, mode=mode)
    timer.save(file_path=timer_file_path, mode=mode)
