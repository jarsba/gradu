import pandas as pd
from napsu_mq.main import create_model

dataset_map = snakemake.config['datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}
datasets = snakemake.input[0]
epsilons = snakemake.config["epsilons"]
MCMC_algorithms = snakemake.config['MCMC_algorithms']

dataset_to_cliques_map = {
    "dummy": [
        ('A', 'B'), ('B', 'C'), ('A', 'C')
    ]
}


def epsilon_str_to_float(epsilon):
    return float(f"{epsilon[0]}.{epsilon[1]}")


for dataset in datasets:
    for epsilon in epsilons:
        for algo in MCMC_algorithms:
            dataframe = pd.read_csv(dataset)
            dataset_name = inverted_dataset_map[dataset]
            n, d = dataframe.shape

            model = create_model(
                input=dataframe,
                dataset_name=dataset_name,
                epsilon=epsilon_str_to_float(epsilon),
                delta=(n ** (-2)),
                cliques=dataset_to_cliques_map[dataset_name],
                MCMC_algo=algo
            )

            print("Writing model to file")
            napsu_result_file = open(f"models/napsu_{dataset}_{epsilon}e_{algo}.dill", "wb")
            model.store(napsu_result_file)
