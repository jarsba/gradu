import dataclasses
from typing import List, Tuple, Union


@dataclasses.dataclass
class JobParameters:
    job_name: str
    experiment_id: str
    dataset: str
    dataset_path: str
    query_list: List[Tuple[str, str]]
    query_string: str
    epsilon: float
    algo: str
    discretization_level: Union[str, None]
    laplace_approximation: bool
    laplace_approximation_algorithm: str
    missing_query: Union[None, str, List[str]]
