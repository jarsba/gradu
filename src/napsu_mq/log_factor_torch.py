import torch
from typing import List, Tuple, Union
from .log_factor import LogFactor


class LogFactorTorch(LogFactor):
    """PyTorch implementation of LogFactor."""

    def tile_values(self, repeat_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        return self.values.repeat(repeat_shape)

    def permute_axes(self, value: torch.Tensor, permutation: Tuple[int, ...]) -> torch.Tensor:
        return value.permute(permutation)

    def logsumexp(self, values: torch.Tensor, axis: Union[int, Tuple[int, ...]]) -> torch.Tensor:
        return torch.logsumexp(values, dim=axis)

    def take_values(self, index: int, axis: int) -> torch.Tensor:
        return torch.select(self.values, axis, index)

    @staticmethod
    def expand_indices(tensor: torch.Tensor, result_shape: Union[torch.Size, List[int]], i: int) -> torch.Tensor:
        shape = result_shape[:i] + result_shape[i + 1:]
        return tensor.expand(shape + [-1]).movedim(-1, i)

    def compute_batch_condition_values(self, values: torch.Tensor, to_remove_index: int,
                                       result_shape: Union[torch.Size, List[int]]) -> torch.Tensor:
        take_indices = torch.zeros(result_shape, dtype=torch.long)
        for i in range(len(self.scope)):
            if i == to_remove_index:
                index_term = self.expand_indices(values, result_shape, 0)
            else:
                index_term = self.expand_indices(torch.arange(self.values.shape[i]), result_shape,
                                                 i - 1 if i > to_remove_index else i)
            mul = 1
            for j in range(i + 1, len(self.scope)):
                mul *= self.values.shape[j]
            take_indices += index_term * mul

        return self.values.take(take_indices)

    def move_values_axis(self, axis: Union[int, Tuple[int, ...]], place: Union[int, Tuple[int, ...]]) -> None:
        self.values = self.values.movedim(axis, place)

    # TODO: check type
    def query(self, queries) -> torch.Tensor:
        result = torch.zeros(len(queries.queries))
        for i, query in enumerate(queries.queries):
            query_permutation = [query.features.index(variable) for variable in self.scope]
            query_value_tuple = query.value_tuple()
            result[i] = self.values[tuple(query_value_tuple[i] for i in query_permutation)]
        return torch.exp(result - self.log_sum_total())
