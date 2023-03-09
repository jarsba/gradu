import sys
import os
from typing import Literal

sys.path.append(snakemake.config['workdir'])

import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
from arviz.data.inference_data import InferenceDataT

from src.utils.path_utils import MODELS_FOLDER
from src.utils.timer import Timer
from src.utils.keygen import get_key
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.query_utils import join_query_list
from src.utils.string_utils import epsilon_str_to_float
from src.utils.data_utils import transform_for_modeling

dataset_map = snakemake.config['original_datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}
dataset_names = [key for key in dataset_map.keys()]
dataset_files = [value for value in dataset_map.values()]
datasets = snakemake.input

print(datasets)

target_files = snakemake.output

epsilons = snakemake.config["epsilons"]
queries = snakemake.config['queries']

storage_file_path = "napsu_experiment_storage_output.csv"
mode: Literal["replace"] = "replace"
timer_file_path = "napsu_MCMC_time_vs_epsilon_comparison.csv"

storage = ExperimentStorage(file_path=storage_file_path, mode=mode)
timer = Timer(file_path=timer_file_path, mode=mode)

algo = "NUTS"

for dataset in datasets:
    for epsilon_str in epsilons:

        epsilon = epsilon_str_to_float(epsilon_str)

        dataset_name = inverted_dataset_map[dataset]

        queries_for_dataset = queries[dataset_name]

        for query_list in queries_for_dataset:
            dataframe = pd.read_csv(dataset)

            dataframe = transform_for_modeling(dataset_name, dataframe)

            n, d = dataframe.shape
            query_str = join_query_list(query_list)
            delta = (n ** (-2))

            experiment_id = get_key()
            experiment_id_ctx.set(experiment_id)

            timer_meta = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "query": query_str,
                "epsilon": epsilon,
                "delta": delta,
                "MCMC_algo": "NUTS",
                "laplace_approximation": True,
                "laplace_approximation_algorithm": "jaxopt_LBFGS"
            }

            dataset_query_str = f"{dataset_name}_{query_str}"

            model_file_path = os.path.join(MODELS_FOLDER, f"napsu_{dataset_query_str}_{epsilon_str}e_{algo}.dill")
            if os.path.exists(model_file_path):
                print(f"Model already exists for {dataset_query_str} query and epsilon {epsilon_str}, skipping")
                continue

            pid = timer.start(f"Main run", **timer_meta)

            print(
                f"PARAMS: \n\tdataset name {dataset_name}\n\tcliques {query_str}\n\tMCMC algo {algo}\n\tepsilon {epsilon_str}\n\tdelta: {delta}\n\tLaplace approximation {True}")

            print("Initializing NapsuMQModel")
            rng = jax.random.PRNGKey(6473286482)

            model = NapsuMQModel()

            result: NapsuMQResult
            inf_data: InferenceDataT

            print(dataframe.dtypes)
            print(dataframe.head())
            print(dataframe.shape)
            print(dataframe.columns)

            result, inf_data = model.fit(
                data=dataframe,
                dataset_name=dataset_name,
                rng=rng,
                epsilon=epsilon,
                delta=delta,
                column_feature_set=query_list,
                MCMC_algo=algo,
                use_laplace_approximation=True,
                return_inference_data=True,
                enable_profiling=True,
                laplace_approximation_algorithm="jaxopt_LBFGS"
            )

            timer.stop(pid)

            model_file_path = os.path.join(MODELS_FOLDER, f"napsu_{dataset_query_str}_{epsilon_str}e_{algo}.dill")
            result.store(model_file_path)

            inf_data.to_netcdf(f"logs/inf_data_{dataset_query_str}_{epsilon_str}e_{algo}.nc")

            # Save storage and timer results every iteration
            storage.save(file_path=storage_file_path, mode=mode)
            timer.save(file_path=timer_file_path, mode=mode)

storage.save(file_path=storage_file_path, mode=mode)
timer.save(file_path=timer_file_path, mode=mode)
