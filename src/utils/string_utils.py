from src.utils.query_utils import join_query_list


def epsilon_str_to_float(epsilon_str: str):
    epsilon = float(f"{epsilon_str[0]}.{epsilon_str[1]}")
    return epsilon


def epsilon_float_to_str(epsilon: float):
    epsilon_str = str(epsilon).replace(".", "")
    return epsilon_str


def format_model_name_string(dataset_name, query_list, epsilon):
    query_str = join_query_list(query_list)
    job_parameter_str = f"{dataset_name}_{epsilon}e_{query_str}"
    return job_parameter_str


def format_model_name_string_with_query_str(dataset_name, query_str, epsilon):
    job_parameter_str = f"{dataset_name}_{epsilon}e_{query_str}"
    return job_parameter_str
