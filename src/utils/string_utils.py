from typing import List, Tuple
from src.utils.query_utils import join_query_list


def epsilon_str_to_float(epsilon_str: str):
    epsilon = float(f"{epsilon_str[0]}.{epsilon_str[1]}")
    return epsilon


def epsilon_float_to_str(epsilon: float):
    epsilon_str = str(epsilon).replace(".", "")
    return epsilon_str


def format_job_parameter_for_napsu_original(dataset_name, query_list, epsilon, algo="NUTS"):
    query_str = join_query_list(query_list)
    job_parameter_str = f"{dataset_name}_{query_str}_{epsilon}e_{algo}"
    return job_parameter_str


def format_job_parameter_for_discretization_datasets(discretization_level, epsilon):
    job_parameter_str = f"{discretization_level}_{epsilon}e"
    return job_parameter_str
