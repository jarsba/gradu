# Originally from https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
# Modified by Authors under the Apache 2.0 license
import itertools
from typing import Iterable, List, Tuple, Any, Callable, Mapping, Union

import networkx as nx
import numpy as np
import scipy.sparse
from disjoint_set import DisjointSet
from mbi import FactoredInference, Dataset, Domain
from scipy import sparse
from scipy.special import logsumexp

from . import privacy_accounting_zcdp as accounting

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""


def check_kwargs(kwargs, name, default_value):
    if name in kwargs:
        return kwargs[name]
    else:
        return default_value


def MST_selection(data: Dataset, epsilon: float, delta: float, cliques_to_include: Iterable[Tuple[str, str]] = [],
                  **kwargs):
    return_MST_weights = check_kwargs(kwargs, "return_MST_weights", False)

    rho = accounting.eps_delta_budget_to_rho_budget(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    if return_MST_weights is True:
        cliques, weights = select(data, rho / 3.0, log1, cliques=cliques_to_include, **kwargs)
        return cliques, weights
    cliques = select(data, rho / 3.0, log1, cliques=cliques_to_include)
    return cliques


def MST(data: Dataset, epsilon: float, delta: float) -> Dataset:
    rho = accounting.eps_delta_budget_to_rho_budget(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, rho / 3.0, log1)
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1 + log2)
    synth = est.synthetic_data()
    return undo_compress_fn(synth)


def measure(data: Dataset, cliques: List, sigma: float, weights: np.ndarray = None) -> List[
    Tuple[scipy.sparse.coo_matrix, np.ndarray, float, List]]:
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


def compress_domain(data: Dataset, measurements: List[Tuple]) -> Tuple[Dataset, List, Callable]:
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3 * sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append((Q, y, sigma, proj))
        else:  # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append((I2, y2, sigma, proj))
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


# TODO: add type to q
def exponential_mechanism(q: Any, eps: float, sensitivity: float, prng=np.random, monotonic=False) -> np.ndarray:
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    print(f"Prbablities: {probas}")
    return prng.choice(q.size, p=probas)


def select(data: Dataset, rho: float, measurement_log: List[Tuple], cliques: Iterable[Tuple[str, str]] = [],
           **kwargs) -> Union[Tuple[List, Mapping], List]:
    return_MST_weights = check_kwargs(kwargs, "return_MST_weights", False)

    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    # Candidates are all pairs of attributes
    candidates = list(itertools.combinations(data.domain.attrs, 2))

    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        # Calculate the L1 distance between the true and estimated data for each pair of attributes
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    print(f"Len weights: {len(weights)}")
    print(f"Weights: {weights}")
    print(f"Sorted weights: {sorted(weights.items(), key=lambda x: x[1])}")

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    print(f"Cliques: {cliques}")
    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        print(f"Candidates: {candidates}")
        print(f"Weights: {wgts}")
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        print(f"Idx: {idx}")
        e = candidates[idx]
        print(f"Edge: {e}")
        T.add_edge(*e)
        ds.union(*e)
        print(f"T edges: {T.edges}")

    print(f"T edges: {T.edges}")

    if return_MST_weights is True:
        return list(T.edges), weights

    return list(T.edges)


def transform_data(data: Dataset, supports: Mapping) -> Dataset:
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def reverse_data(data: Dataset, supports: Mapping) -> Dataset:
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)
