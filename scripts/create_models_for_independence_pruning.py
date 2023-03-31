import sys
from typing import Literal
import pickle

sys.path.append(snakemake.config['workdir'])

from jax.config import config

config.update("jax_enable_x64", True)

import torch

torch.set_default_dtype(torch.float64)

from arviz.data.inference_data import InferenceDataT
import pandas as pd

from src.utils.timer import Timer
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.data_utils import transform_for_modeling
from src.utils.seed_utils import set_seed
from src.utils.job_parameters import JobParameters
from src.utils.string_utils import epsilon_float_to_str

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""

Steps:

Start with Adult dataset with 6 variables and determine the independence ranking for each pair (15 combinations) using 
Chi-Square test. Run Napsu-MQ with full set of marginals and with each full set with one pair removed. Look at the 
downstream results like confidence interval coverage and compare the predicted result with the real results. Continue 
with 5 variables using the set with of variables with least harm to downstream accuracy and repeat for 5 and 4 variables. 

"""

if __name__ == "__main__":
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
    missing_query = parameter_combinations.missing_query

    target_file = str(snakemake.output[0])

    adult_dataset = pd.read_csv(dataset_path)
    adult_train_df = transform_for_modeling("adult_small", adult_dataset)

    storage_file_path = "napsu_independence_pruning_storage.csv"
    mode: Literal["append"] = "append"
    timer_file_path = "napsu_independence_pruning_timer.csv"

    storage = ExperimentStorage(file_path=storage_file_path, mode=mode)
    timer = Timer(file_path=timer_file_path, mode=mode)

    n, d = adult_train_df.shape
    delta = (n ** (-2))

    timer_meta = {
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "query": query_str,
        "missing_query": missing_query,
        "epsilon": epsilon,
        "delta": delta,
        "MCMC_algo": algo,
        "laplace_approximation": laplace_approximation,
        "laplace_approximation_algorithm": laplace_approximation_algorithm
    }

    pid = timer.start(f"Main run", **timer_meta)

    print(
        f"PARAMS: \n\tdataset name {dataset_name}\n\tmissing query {missing_query}\n\tMCMC algo: NUTS\n\tepsilon {epsilon_str}\n\tdelta: {delta}\n\tLaplace approximation {laplace_approximation}")

    print("Initializing NapsuMQModel")

    model = NapsuMQModel()

    result: NapsuMQResult
    inf_data: InferenceDataT

    result, inf_data = model.fit(
        data=adult_train_df,
        dataset_name=dataset_name,
        rng=rng,
        epsilon=epsilon,
        delta=delta,
        column_feature_set=query_list,
        MCMC_algo=algo,
        use_laplace_approximation=laplace_approximation,
        return_inference_data=True,
        missing_query=missing_query,
        enable_profiling=False,
        laplace_approximation_algorithm=laplace_approximation_algorithm,
        laplace_approximation_forward_mode=True
    )

    timer.stop(pid)

    print("Writing model to file")
    result.store(target_file)

    inf_data.to_netcdf(f"logs/inf_data_independence_pruning_{dataset_name}_{epsilon}e_{missing_query}.nc")

    storage.save(file_path=storage_file_path, mode=mode)
    timer.save(file_path=timer_file_path, mode=mode)
