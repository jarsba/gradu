from src.utils.query_utils import join_query_list
import itertools
from src.utils.preprocess_dataset import ADULT_COLUMNS_SMALL
from src.utils.string_utils import format_job_parameter_for_napsu_original


def get_query_list(queries, dataset_name):
    query_strings = []
    dataset_queries = queries[dataset_name]
    for query_list in dataset_queries:
        query_str = join_query_list(query_list)
        query_strings.append(query_str)

    return query_strings


def generate_napsu_original_products(datasets, queries, epsilons, MCMC_algorithms):
    products = []

    for dataset in datasets:
        dataset_queries = queries[dataset]
        for query_list in dataset_queries:
            for epsilon in epsilons:
                for algo in MCMC_algorithms:
                    product_str = format_job_parameter_for_napsu_original(dataset, query_list, epsilon, algo)
                    products.append(product_str)

    return products


def query_dataset_product(datasets, queries):
    query_dataset_product = []

    for dataset in datasets:
        dataset_queries = queries[dataset]
        for query_list in dataset_queries:
            query_str = join_query_list(query_list)
            dataset_query = f"{dataset}_{query_str}"
            query_dataset_product.append(dataset_query)

    return query_dataset_product


def query_independence_pruning_product():
    adult_small_columns = ADULT_COLUMNS_SMALL

    missing_queries = []
    marginal_pairs = list(itertools.combinations(adult_small_columns, 2))
    marginal_pair_strings = [f"{pair[0]}+{pair[1]}" for pair in marginal_pairs]
    missing_queries.extend(marginal_pair_strings)
    missing_queries.append('all')
    missing_queries.append('none')

    return missing_queries


def generate_products(queries_dataset_product, epsilons, MCMC_algorithms):
    products = []
    for query_dataset in queries_dataset_product:
        for epsilon in epsilons:
            for algo in MCMC_algorithms:
                product_str = f"{query_dataset}_{epsilon}e_{algo}"
                products.append(product_str)

    return products


def generate_dicretization_product(epsilons, discretization_levels):
    products = []
    for epsilon in epsilons:
        for discretization_level in discretization_levels:
            product_str = f"{discretization_level}_{epsilon}e"
            products.append(product_str)

    return products


def generate_independence_pruning_products(epsilons, queries):
    products = []
    for epsilon in epsilons:
        for query in queries:
            product_str = f"{query}_missing_{epsilon}e"
            products.append(product_str)

    return products
