# %%
import json

import tensorflow as tf

import tensorflow_probability as tfp
from data import load_data, prepare_data
from objective import penalty


# %%
@tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """Since this is bulk of the computation, using @tf.function
    here to compile a static graph for tfp.mcmc.sample_chain significantly improves
    performance, especially when enabling XLA (Accelerated Linear Algebra).
    https://tensorflow.org/xla#explicit_compilation_with_tffunction
    https://github.com/tensorflow/probability/issues/728#issuecomment-573704750
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)


# %%
mols_atoms, coords, charges, titles, reference = load_data()
ref_energies = reference.iloc[:, 1].tolist()

with open("../parameters/parameters-pm3.json") as file:
    start_params = json.loads(file.read())

# param_keys needed for mndo.set_params
# param_values acts as initial condition for HMC kernel
param_keys, param_values = prepare_data(mols_atoms, start_params)

# %%
def target_log_prob_fn(param_vals):
    return -penalty(param_vals, param_keys, ref_energies, "_tmp_optimizer")


# %%
step_size = 1e-3
kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
adapt_steps = 5000
adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    kernel,
    num_adaptation_steps=adapt_steps,
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
        step_size=new_step_size
    ),
    step_size_getter_fn=lambda pkr: pkr.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
)
chain, trace, final_kernel_results = sample_chain(
    kernel=adaptive_kernel,
    num_results=2000,
    current_state=param_values,
    return_final_kernel_results=True,
)


# %%
