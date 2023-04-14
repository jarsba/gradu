from typing import Tuple, List

from src.utils.query_utils import join_query_list
import itertools
from src.utils.preprocess_dataset import ADULT_COLUMNS_SMALL, ADULT_COLUMNS_INDEPENDENCE_PRUNING
from src.utils.string_utils import epsilon_float_to_str


def get_query_list(queries, dataset_name) -> List[str]:
    query_strings = []
    dataset_queries = queries[dataset_name]
    for query_list in dataset_queries:
        query_str = join_query_list(query_list)
        query_strings.append(query_str)

    return query_strings


def generate_dataset_query_epsilon_products(datasets, queries, epsilons) -> Tuple[List, List, List]:
    dataset_list = []
    query_list = []
    epsilon_list = []

    for dataset in datasets:
        dataset_queries = get_query_list(queries, dataset)
        for query_set in dataset_queries:
            for epsilon in epsilons:
                epsilon_str = epsilon_float_to_str(epsilon)
                dataset_list.append(dataset)
                query_list.append(query_set)
                epsilon_list.append(epsilon_str)

    return dataset_list, query_list, epsilon_list


def generate_dataset_query_epsilon_products_independence_queries(datasets, queries, epsilons) -> Tuple[
    List, List, List]:
    dataset_list = []
    query_list = []
    epsilon_list = []

    for dataset in datasets:
        for query_set in queries:
            for epsilon in epsilons:
                epsilon_str = epsilon_float_to_str(epsilon)
                dataset_list.append(dataset)
                query_list.append(query_set)
                epsilon_list.append(epsilon_str)

    return dataset_list, query_list, epsilon_list


def generate_independence_pruning_missing_queries():
    """Generate missing query strings for independence pruning datasets"""
    adult_independence_pruning_columns = ADULT_COLUMNS_INDEPENDENCE_PRUNING
    adult_independence_pruning_columns = sorted(adult_independence_pruning_columns)

    independence_pruning_missing_query_strings = []

    marginal_pairs = list(itertools.combinations(adult_independence_pruning_columns, 2))

    marginal_pair_strings = [f"{pair[0]}+{pair[1]}" for pair in marginal_pairs]

    independence_pruning_missing_query_strings.extend(marginal_pair_strings)
    independence_pruning_missing_query_strings.append('all')
    independence_pruning_missing_query_strings.append('none')

    return independence_pruning_missing_query_strings


def generate_linear_regression_products(epsilons, n_repeats=20) -> Tuple[List, List, List]:
    dataset_list = []
    epsilon_list = []
    repeat_index_list = []
    for i in range(n_repeats):
        for n_categories in range(2, 8):
            for epsilon in epsilons:
                epsilon_str = epsilon_float_to_str(epsilon)
                dataset_name = f"dummy_5x{n_categories}"

                dataset_list.append(dataset_name)
                epsilon_list.append(epsilon_str)
                repeat_index_list.append(i)

    return dataset_list, epsilon_list, repeat_index_list
