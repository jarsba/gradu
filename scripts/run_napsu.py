import pandas as pd
from src.napsu_mq.main import create_model
from src.utils.timer import Timer
from src.utils.keygen import get_key
from src.utils.experiment_storage import ExperimentStorage
import contextvars

dataset_map = snakemake.config['datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}
dataset_names = [key for key in dataset_map.keys()]
dataset_files = [value for value in dataset_map.values()]
datasets = snakemake.input[0]
epsilons = snakemake.config["epsilons"]
MCMC_algorithms = snakemake.config['MCMC_algorithms']
queries = snakemake.config['queries']

storage = ExperimentStorage()
timer = Timer()
experiment_id_ctx = contextvars.ContextVar('experiment_id')


def epsilon_str_to_float(epsilon):
    return float(f"{epsilon[0]}.{epsilon[1]}")


for dataset in datasets:
    for epsilon in epsilons:
        for algo in MCMC_algorithms:

            queries_for_dataset = queries[dataset]

            for query in queries_for_dataset:
                dataframe = pd.read_csv(dataset)
                dataset_name = inverted_dataset_map[dataset]
                n, d = dataframe.shape
                query_str = "".join(query)

                experiment_id = get_key()
                experiment_id_ctx.set(experiment_id)

                timer_meta = {
                    "experiment_id": experiment_id,
                    "dataset_name": dataset_name,
                    "query": query_str,
                    "epsilon": epsilon,
                    "delta": (n ** (-2)),
                    "MCMC_algo": "NUTS",
                    "laplace_approximation": True
                }

                pid = timer.start(f"Main run", **timer_meta)

                model = create_model(
                    input=dataframe,
                    dataset_name=dataset_name,
                    epsilon=epsilon_str_to_float(epsilon),
                    delta=(n ** (-2)),
                    cliques=query,
                    MCMC_algo=algo,
                )

                timer.stop(pid)

                dataset_query_str = f"{dataset}_{query_str}"

                print("Writing model to file")
                napsu_result_file = open(f"models/napsu_{experiment_id}_{dataset_query_str}_{epsilon}e_{algo}.dill",
                                         "wb")
                model.store(napsu_result_file)

timer.to_csv("napsu_MCMC_time_vs_epsilon_comparison.csv")
storage.to_csv("napsu_experiment_storage_output.csv")
