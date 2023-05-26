import sys

sys.path.append("/home/local/jarlehti/projects/gradu")
sys.path.append("/wrk-vakka/users/jarlehti/gradu")

import jax.numpy as jnp
import numpy as np
import pandas as pd
import d3p
from twinify.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from scripts.base_ci_coverage import calculate_ci_coverage_objects
from src.utils.preprocess_dataset import get_binary3d_train
from src.napsu_mq.binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator

rng = d3p.random.PRNGKey(42526709)
data_gen_rng, inference_rng = d3p.random.split(rng)
data_gen_rng = d3p.random.convert_to_jax_rng_key(data_gen_rng)

binary3d = get_binary3d_train()

coefficients = [-2.0, 4.0]
dataset = BinaryLogisticRegressionDataGenerator(jnp.array(coefficients)).generate_data(n=100000, rng_key=data_gen_rng)
orig_df = pd.DataFrame(dataset, dtype="category")
n, d = orig_df.shape

orig_df_np = orig_df.to_numpy(dtype="int8")

rng = d3p.random.PRNGKey(74249069)
inference_rng, sampling_rng = d3p.random.split(inference_rng)

required_marginals = [(0, 2), (1, 2)]
epsilon = 1

# We can define column marginal relationships that we want to preserve
model = NapsuMQModel(column_feature_set=required_marginals, use_laplace_approximation=True)
model = model.fit(
    data=orig_df,
    rng=inference_rng,
    epsilon=epsilon,
    delta=(n ** (-2))
)

target_column_index = d - 1

meta = {
    'epsilon': epsilon
}

ci_intervals = np.round(np.linspace(0.05, 0.95, 19), 2)
ci_df = calculate_ci_coverage_objects(model, orig_df_np, meta, rng=sampling_rng, confidence_intervals=ci_intervals,
                                      n_repeats=50, n_datasets=100, n_syn_datapoints=n,
                                      target_column_index=target_column_index)

ci_df.to_csv("twinify_confidence_interval_calibration.csv")
