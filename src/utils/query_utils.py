from typing import Mapping, Tuple, Union, List
from src.napsu_mq.marginal_query import QueryList


def calculate_query_number(canonical_query_set: Mapping[Tuple, QueryList]):
    total_queries = 0

    for key, value in canonical_query_set.items():
        n_queries = len(value.queries)
        total_queries += n_queries

    return total_queries


def join_query_list(query_list: List[Tuple[str, str]]) -> str:
    query_str = ""

    if len(query_list) == 0:
        return "empty"

    for i, tuple in enumerate(query_list):
        marginal_str = "".join(tuple)
        if i == len(query_list) - 1:
            query_str += f"{marginal_str}"
        else:
            query_str += f"{marginal_str}+"

    return query_str
