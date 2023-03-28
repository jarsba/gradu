import itertools
import sys
import os

sys.path.append(snakemake.config['workdir'])

import pickle
import pandas as pd
from src.utils.keygen import generate_experiment_id
from src.utils.query_utils import join_query_list
from src.utils.path_utils import DATA_FOLDER
from src.utils.job_parameters import JobParameters
from src.utils.string_utils import epsilon_float_to_str

PARAMETER_COMBINATIONS_FOLDER = os.path.join(DATA_FOLDER, "parameter_combinations")

if __name__ == "__main__":

    epsilons = snakemake.config["epsilons"]
    queries = snakemake.config['queries']
    discretization_levels = ['low', 'high']
    algo = "NUTS"

    dataset = snakemake.config['datasets']
    dataset_names = [key for key in dataset.keys()]
    dataset_files = [value for value in dataset.values()]

    # Create JobParameter objects for original NAPSU-MQ models
    original_datasets = snakemake.config['original_datasets']

    for dataset_name, dataset_path in original_datasets.items():
        for epsilon in epsilons:
            for query_list in queries[dataset_name]:
                experiment_id = generate_experiment_id()
                query_str = join_query_list(query_list)

                epsilon_str = epsilon_float_to_str(epsilon)

                job_name = f"napsu_original_model_parameters_{dataset_name}_{epsilon_str}e_{query_str}"
                job_parameter = JobParameters(
                    job_name=job_name,
                    experiment_id=experiment_id,
                    dataset=dataset_name,
                    dataset_path=dataset_path,
                    query_list=query_list,
                    query_string=query_str,
                    epsilon=epsilon,
                    algo=algo,
                    discretization_level=None,
                    laplace_approximation=True,
                    laplace_approximation_algorithm="torch_LBFGS",
                    missing_query=None
                )
                with open(os.path.join(PARAMETER_COMBINATIONS_FOLDER, f"{job_name}.pickle"), 'wb') as f:
                    pickle.dump(job_parameter, f)

    # Create JobParameter objects for discretization models

    discretization_datasets = snakemake.config['discretization_datasets']

    for dataset_name, dataset_path in discretization_datasets.items():
        for epsilon in epsilons:
            for query_list in queries[dataset_name]:
                experiment_id = generate_experiment_id()
                query_str = join_query_list(query_list)
                discretization_level = "low" if "low" in dataset_name else "high"
                epsilon_str = epsilon_float_to_str(epsilon)

                job_name = f"napsu_discretization_model_parameters_{dataset_name}_{epsilon_str}e_{query_str}"
                job_parameter = JobParameters(
                    job_name=job_name,
                    experiment_id=experiment_id,
                    dataset=dataset_name,
                    dataset_path=dataset_path,
                    query_list=query_list,
                    query_string=query_str,
                    epsilon=epsilon,
                    algo=algo,
                    discretization_level=discretization_level,
                    laplace_approximation=True,
                    laplace_approximation_algorithm="torch_LBFGS",
                    missing_query=None
                )
                with open(os.path.join(PARAMETER_COMBINATIONS_FOLDER, f"{job_name}.pickle"), 'wb') as f:
                    pickle.dump(job_parameter, f)

    independence_pruning_datasets = snakemake.config['independence_pruning_datasets']

    for dataset_name, dataset_path in independence_pruning_datasets.items():
        dataset = pd.read_csv(dataset_path)
        dataset_columns = list(dataset.columns)
        dataset_columns = sorted(dataset_columns)

        # Make set of sets
        marginal_pairs = list(itertools.combinations(dataset_columns, 2))
        full_set_of_marginals = marginal_pairs

        # Removes set from list of tuples that are interpreted as list of sets
        immutable_set_remove = lambda element, list_obj: list(filter(lambda x: set(x) != set(element), list_obj))

        # List of lists with tuples of all 2-way marginals
        test_queries = [immutable_set_remove(pair, full_set_of_marginals) for pair in marginal_pairs]
        # Add also full and no queries
        test_queries.append(full_set_of_marginals)
        test_queries.append([])

        for epsilon in epsilons:
            for query_list in test_queries:
                experiment_id = generate_experiment_id()
                query_str = join_query_list(query_list)
                epsilon_str = epsilon_float_to_str(epsilon)

                if len(query_list) == 0:
                    query_removed = full_set_of_marginals
                    missing_query = "all"
                elif len(query_list) == len(full_set_of_marginals):
                    query_removed = []
                    missing_query = "none"
                else:
                    query_removed = list(set(full_set_of_marginals) - set(query_list))
                    missing_query = [f"{pair[0]}+{pair[1]}" for pair in query_removed][0]

                job_name = f"napsu_independence_pruning_model_parameters_{dataset_name}_{epsilon_str}e_{missing_query}"
                job_parameter = JobParameters(
                    job_name=job_name,
                    experiment_id=experiment_id,
                    dataset=dataset_name,
                    dataset_path=dataset_path,
                    query_list=query_list,
                    query_string=query_str,
                    epsilon=epsilon,
                    algo=algo,
                    discretization_level=None,
                    laplace_approximation=True,
                    laplace_approximation_algorithm="torch_LBFGS",
                    missing_query=missing_query
                )
                with open(os.path.join(PARAMETER_COMBINATIONS_FOLDER, f"{job_name}.pickle"), 'wb') as f:
                    pickle.dump(job_parameter, f)
