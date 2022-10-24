from dataclasses import dataclass
import numpy as np


@dataclass
class SynthDataObject:
    synth_data: np.ndarray
    experiment_id: str
    n_datasets: int
    n_rows: int
    n_cols: int
    original_dataset: str
    query: str
    inference_algorithm: str
    epsilon: float
    delta: float