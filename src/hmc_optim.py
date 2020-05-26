# %%
import json
import os
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import mndo
from data import load_data, prepare_data
from objective import jacobian_parallel, penalty
from hmc_utils import (
    sample_chain,
    trace_fn_nuts,
    get_nuts_kernel,
    trace_fn_hmc,
    get_hmc_kernel,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %%
# Load the reference data
mols_atoms, mols_coords, charges, titles, reference = load_data(query_size=100)
ref_energies = reference.iloc[:, 1].tolist()

# NOTE we should refactor so that we don't need _tmp_molecules here
tmp_molecule_file = "_tmp_molecules"
mndo_method = "MNDO"
mndo.write_tmp_optimizer(
    atoms=mols_atoms, coords=mols_coords, filename=tmp_molecule_file, method=mndo_method
)

mndo_input = mndo.get_inputs(
    atoms_list=mols_atoms,
    coords_list=mols_coords,
    charges=np.zeros_like(mols_atoms),
    titles=range(len(mols_atoms)),
    method=mndo_method,
)

# %%
@tf.custom_gradient
def target_log_prob_fn(*param_vals):
    log_likelihood = -penalty(
        param_vals, param_keys, filename=tmp_molecule_file, ref_props=ref_energies
    )

    def grad_fn(*dys):
        grad = jacobian_parallel(
            param_vals,
            mndo_input=mndo_input,
            param_keys=param_keys,
            dh=1e-5,
            # filename=tmp_molecule_file,
            ref_props=ref_energies,
        )
        return grad

    return log_likelihood, grad_fn


def real_target_log_prob_fn(*param_vals):
    res = tf.py_function(target_log_prob_fn, inp=param_vals, Tout=tf.float64)
    # Avoid tripping up sample_chain due to loss of output shape in tf.py_function
    # when used in a tf.function context. https://tinyurl.com/y9ttqdpt
    res.set_shape(param_vals[0].shape[:-1])  # assumes parameter is vector-valued
    return res


# %%


# %%
# Find param_keys for an example set of parameters
with open("../parameters/parameters-mndo-mean.json") as file:
    start_params = json.loads(file.read())

# param_keys needed for mndo.set_params
# param_values acts as initial condition for HMC kernel
param_keys, param_values = prepare_data(mols_atoms, start_params)

# %%
step_size = tf.cast(1e-3, tf.float64)
n_adapt_steps = 100
# param_values = [tf.constant(x) for x in param_values]
param_values = [tf.random.truncated_normal([], stddev=1.0) for _ in param_keys]

# %%
# with tf.GradientTape() as tape:
#     tape.watch(param_values)
#     lp = real_target_log_prob_fn(*param_values)
#     print(tape.gradient(lp, param_values))

# %%
now = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_dir = f"runs/hmc-trace/{now}"
summary_writer = tf.summary.create_file_writer(log_dir)

# %%
# chain, trace, final_kernel_results = sample_chain(
#     num_results=30,
#     current_state=param_values,
#     kernel=get_nuts_kernel(real_target_log_prob_fn, step_size, n_adapt_steps),
#     return_final_kernel_results=True,
#     trace_fn=partial(trace_fn_nuts, summary_writer=summary_writer),
# )

chain, trace, final_kernel_results = sample_chain(
    num_results=30,
    current_state=param_values,
    kernel=get_hmc_kernel(real_target_log_prob_fn, step_size, n_adapt_steps),
    return_final_kernel_results=True,
    trace_fn=partial(trace_fn_hmc, summary_writer=summary_writer),
)

with open("../parameters/parameters-opt-hmc.json", "w") as f:
    json.dump([list(x) for x in chain], f)

# # %%
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].hist(chain)
# axs[0, 0].set_title("chain histogram")
# axs[0, 1].plot(chain)
# axs[0, 1].set_title("chain plot")
# axs[1, 0].hist(trace)
# axs[1, 0].set_title("trace histogram")
# axs[1, 1].plot(trace)
# axs[1, 1].set_title("trace plot")
