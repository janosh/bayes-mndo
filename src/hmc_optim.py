# %%
import json
import os
import pathlib
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf
from chemhelp import mndo, units

from data import load_data, prepare_params
from hmc_utils import get_nuts_kernel, sample_chain, trace_fn_nuts
from objective import jacobian_parallel, penalty

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mols_atoms, mols_coords, _, _, reference = load_data(query_size=100, offset=110)
ref_energies = reference["binding_energy"].values

# Switch from Hartree to KCal/Mol
ref_energies *= units.hartree_to_kcalmol

dh = 1e-5
n_procs = 2
method = "MNDO"

# NOTE we probably can refactor to remove the duplication of input files
filename = "_tmp_molecules"
scrdir = "_tmp_optim"

pathlib.Path(scrdir).mkdir(parents=True, exist_ok=True)

# TODO JCK At some point we need to evaluate non-zero molecules
n_molecules = len(mols_atoms)
mols_charges = np.zeros(n_molecules)
mols_names = np.arange(n_molecules)

mndo.write_input_file(
    mols_atoms,
    mols_coords,
    mols_charges,
    mols_names,
    method,
    os.path.join(scrdir, filename),
    read_params=True,
)

# %%
with open("parameters/parameters-mndo-mean.json", "r") as f:
    mean_params = json.loads(f.read())

with open("parameters/parameters-mndo-std.json", "r") as f:
    scale_params = json.loads(f.read())

param_keys, _ = prepare_params(mols_atoms, mean_params)
param_values = [tf.random.truncated_normal([], stddev=1.0) for _ in param_keys]

root = os.path.abspath(__file__).split("/src", 1)[0]

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "n_procs": n_procs,
    "dh": dh,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "binary": root + "/mndo/mndo99_binary",
    "scr": scrdir,
}

# %%
@tf.custom_gradient
def target_log_prob_fn(*param_vals):
    log_likelihood = -penalty(param_vals, **kwargs)

    def grad_fn(*dys):
        # grad = jacobian(param_vals, **kwargs)
        grad = jacobian_parallel(param_vals, **kwargs)
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

with open("parameters/parameters-opt-hmc.json", "w") as f:
    json.dump([list(x) for x in chain], f)
