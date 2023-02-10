from dataclasses import dataclass


@dataclass
class ConfidenceIntervalObject:
    original_dataset_name: str
    index: int
    n_datasets: int
    conf_int_range: float
    conf_int_start: float
    conf_int_end: float
    conf_int_width: float
    true_parameter_value: float
    contains_true_parameter: bool
    # Contains any additional information that we want to store
    meta: dict

    def to_dict(self):
        return {
            'original_dataset_name': self.original_dataset_name,
            'index': self.index,
            'n_datasets': self.n_datasets,
            'conf_int_range': self.conf_int_range,
            'conf_int_start': self.conf_int_start,
            'conf_int_end': self.conf_int_end,
            'conf_int_width': self.conf_int_width,
            'true_parameter_value': self.true_parameter_value,
            'contains_true_parameter': self.contains_true_parameter,
            **self.meta
        }