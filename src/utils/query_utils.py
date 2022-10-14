from typing import Mapping
from src.napsu_mq.marginal_query import QueryList


def calculate_query_number(canonical_query_set: Mapping[QueryList]):
    total_queries = 0

    for key, value in canonical_query_set.items():
        n_queries = len(value.queries)
        total_queries += n_queries

    return total_queries
