# %%
import json
import os
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf

import mndo
from data import load_data, prepare_data
from objective import jacobian, jacobian_parallel, penalty
from hmc_utils import sample_chain, trace_fn_nuts, get_nuts_kernel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %%
# Load the reference data
mols_atoms, mols_coords, charges, titles, reference = load_data(query_size=100)
ref_energies = reference["binding_energy"].tolist()

# NOTE we should refactor so that we don't need _tmp_molecules here
filename = "_tmp_molecules"
mndo_method = "MNDO"
mndo.write_tmp_optimizer(
    atoms=mols_atoms, coords=mols_coords, filename=filename, method=mndo_method
)

# %%
# Find param_keys for an example set of parameters
# with open("../parameters/parameters-mndo-mean.json") as file:
#     start_params = json.loads(file.read())

with open("../parameters/parameters-mndo-mean.json", "r") as file:
    raw_json = file.read()
    mean_params = json.loads(raw_json)

with open("../parameters/parameters-mndo-std.json", "r") as file:
    raw_json = file.read()
    scale_params = json.loads(raw_json)

# param_keys, param_values = prepare_data(mols_atoms, start_params)
param_keys, _ = prepare_data(mols_atoms, mean_params)
# param_values = [tf.constant(x) for x in param_values]
param_values = [tf.random.truncated_normal([], stddev=1.0) for _ in param_keys]

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "n_procs": 2,
    "binary": "/home/reag2/PhD/second-year/bayes-mndo/mndo/mndo99_binary",
}

# %%
@tf.custom_gradient
def target_log_prob_fn(*param_vals):
    log_likelihood = -penalty(param_vals, **kwargs)

    def grad_fn(*dys):
        # grad = jacobian(param_vals, dh=1e-5, **kwargs)
        grad = jacobian_parallel(param_vals, dh=1e-5, **kwargs)
        return grad.tolist()

    return log_likelihood, grad_fn


def real_target_log_prob_fn(*param_vals):
    res = tf.py_function(target_log_prob_fn, inp=param_vals, Tout=tf.float64)
    # Avoid tripping up sample_chain due to loss of output shape in tf.py_function
    # when used in a tf.function context. https://tinyurl.com/y9ttqdpt
    res.set_shape(param_vals[0].shape[:-1])  # assumes parameter is vector-valued
    return res


# %%
step_size = tf.cast(5e-3, tf.float64)
n_adapt_steps = 100

# with tf.GradientTape() as tape:
#     tape.watch(param_values)
#     lp = real_target_log_prob_fn(*param_values)
#     print(tape.gradient(lp, param_values))

# %%
now = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_dir = f"runs/hmc-mndo/{now}"
summary_writer = tf.summary.create_file_writer(log_dir)

# %%
chain, trace, final_kernel_results = sample_chain(
    num_results=30,
    current_state=param_values,
    kernel=get_nuts_kernel(real_target_log_prob_fn, step_size, n_adapt_steps),
    return_final_kernel_results=True,
    trace_fn=partial(trace_fn_nuts, summary_writer=summary_writer),
)

with open("../parameters/parameters-opt-hmc.json", "w") as f:
    json.dump([list(x) for x in chain], f)
