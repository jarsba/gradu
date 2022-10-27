import sys

sys.path.append(snakemake.config['workdir'])

import jax
import pandas as pd
from arviz.data.inference_data import InferenceDataT

from src.utils.timer import Timer
from src.utils.keygen import get_hash, get_key
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult

dataset_map = snakemake.config['datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}
dataset_names = [key for key in dataset_map.keys()]
dataset_files = [value for value in dataset_map.values()]
datasets = snakemake.input

target_files = snakemake.output

epsilons = snakemake.config["epsilons"]
MCMC_algorithms = snakemake.config['MCMC_algorithms']
queries = snakemake.config['queries']

storage = ExperimentStorage()
timer = Timer()


def epsilon_str_to_float(epsilon):
    return float(f"{epsilon[0]}.{epsilon[1]}")


for dataset in datasets:
    for epsilon_str in epsilons:

        epsilon = epsilon_str_to_float(epsilon_str)

        for algo in MCMC_algorithms:

            dataset_name = inverted_dataset_map[dataset]

            queries_for_dataset = queries[dataset_name]

            for query in queries_for_dataset:
                dataframe = pd.read_csv(dataset)
                n, d = dataframe.shape
                query_str = "".join(query)
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
                    "laplace_approximation": True
                }

                pid = timer.start(f"Main run", **timer_meta)

                print(
                    f"PARAMS: \n\tdataset name {dataset_name}\n\tcliques {''.join(query)}\n\tMCMC algo {algo}\n\tepsilon {epsilon_str}\n\tdelta: {delta}\n\tLaplace approximation {True}")

                print("Initializing NapsuMQModel")
                rng = jax.random.PRNGKey(6473286482)

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
                    use_laplace_approximation=True,
                    return_inference_data=True
                )

                timer.stop(pid)

                dataset_query_str = f"{dataset}_{query_str}"

                print("Writing model to file")
                napsu_result_file = open(f"models/napsu_{dataset_query_str}_{epsilon_str}e_{algo}.dill",
                                         "wb")
                result.store(napsu_result_file)

                inf_data.to_netcdf(f"logs/inf_data_{dataset_query_str}_{epsilon_str}e_{algo}.nc")

timer.to_csv("napsu_MCMC_time_vs_epsilon_comparison.csv")
storage.to_csv("napsu_experiment_storage_output.csv")
