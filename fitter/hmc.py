# %%
import json

import tensorflow as tf

import tensorflow_probability as tfp
from data import load_data, prepare_data
from objective import penalty


# %%
# @tf.function(experimental_compile=True)
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
dist = tfp.distributions.Normal(0, 1)


def target_log_prob_fn(*param_vals):
    # log_likelihood = -penalty(param_vals, param_keys, ref_energies, "_tmp_optimizer")
    # print("log_likelihood:", log_likelihood)
    print("param_vals:", param_vals)
    # return log_likelihood
    return dist.log_prob(*param_vals)


# %%
log_dir = "runs/hmc-trace/"
summary_writer = tf.summary.create_file_writer(log_dir)


def trace_fn(current_state, kernel_results, summary_freq=10, callbacks=[]):
    """Can be passed to the HMC kernel to obtain a trace of intermediate
    kernel results and histograms of the network parameters in Tensorboard.
    """
    step = kernel_results.step
    with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
        print("kernel_results:", kernel_results)
        tf.summary.scalar("kernel_results", kernel_results, step=step)
        tf.summary.flush(writer=summary_writer)
        return kernel_results, [cb(*current_state) for cb in callbacks]


# %%
step_size = 1e-3
kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
adapt_steps = 400
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
    num_results=100,
    current_state=tf.constant([2.0, 2.0]),
    return_final_kernel_results=True,
    trace_fn=trace_fn,
)

# with summary_writer.as_default():
#     tf.summary.trace_export(name="hmc_trace", step=final_kernel_results.step)
# summary_writer.close()

# %%
