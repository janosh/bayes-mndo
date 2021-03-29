# %%
import os
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from matplotlib import cm

# Axes3D import has side effects, it enables using projection='3d' in add_subplot
from mpl_toolkits.mplot3d import Axes3D  # noqa

from bo_bench import (
    branin_hoo_factory,
    branin_hoo_fn,
    branin_hoo_params,
    sample_branin_hoo,
)
from hmc_utils import get_nuts_kernel, sample_chain, trace_fn_nuts

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %%
# Generate random data set
xy, z_true = sample_branin_hoo(100)


def penalty(*params):
    z_pred = branin_hoo_factory(*params)(xy)
    # Normally we'd just return -tf.metrics.mse(z_true, z_pred). But to test if
    # custom gradients are the reason HMC isn't accepting steps on MNDO, we
    # explicitly avoid autodiff.
    se = (z_true - z_pred) ** 2
    return tf.reduce_mean(se, axis=-1)


def jacobian(*params, dh=1e-5):
    """
    Args:
        params: values for each Branin-Hoo param
        dh: small value for numerical gradients
    """
    params = list(params)
    grad = np.zeros_like(params)

    for i, _ in enumerate(params):
        params[i] += dh
        forward = penalty(*params)

        params[i] -= 2 * dh
        backward = penalty(*params)

        de = forward - backward
        grad[i] = -de / (2 * dh)

        params[i] += dh  # undo in-place changes to params for next iteration
    return grad


# %%
@tf.custom_gradient
def custom_grad_target_log_prob_fn(*params):
    log_likelihood = -penalty(*params)

    def grad_fn(*dys):
        grad = jacobian(*params)
        return grad.tolist()

    return log_likelihood, grad_fn


def target_log_prob_fn(*params):
    res = tf.py_function(custom_grad_target_log_prob_fn, inp=params, Tout=tf.float64)
    # Avoid tripping up sample_chain due to loss of output shape in tf.py_function
    # when used in a tf.function context. https://tinyurl.com/y9ttqdpt
    res.set_shape(params[0].shape[:-1])  # assumes parameter is vector-valued
    return res


def target_log_prob_fn_autodiff(param_vals):
    z_pred = branin_hoo_factory(*param_vals)(xy)
    return -tf.metrics.mse(z_true, z_pred)


# %%
# Casting step_size and init_state needed due to TFP bug
# https://github.com/tensorflow/probability/issues/904#issuecomment-624272845
step_size = tf.cast(1e-3, tf.float64)
bh_params_2x = [2 * param for param in branin_hoo_params.values()]
init_state = tf.constant(bh_params_2x, tf.float64)
n_adapt_steps = 200

# with tf.GradientTape() as tape:
#     tape.watch(init_state)
#     lp = target_log_prob_fn(*init_state)
#     print(tape.gradient(lp, init_state))

# %%
now = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_dir = f"runs/hmc-test/{now}"
summary_writer = tf.summary.create_file_writer(log_dir)

chain, trace, final_kernel_results = sample_chain(
    num_results=100,
    current_state=init_state,
    kernel=get_nuts_kernel(target_log_prob_fn, step_size, n_adapt_steps),
    return_final_kernel_results=True,
    trace_fn=partial(trace_fn_nuts, summary_writer=summary_writer),
)
burnin, samples = chain[:n_adapt_steps], chain[n_adapt_steps:]


# %%
# Plot the Branin-Hoo surface
xr = np.linspace(-5, 15, 21)
yr = np.linspace(0, 10, 11)
XY = np.meshgrid(xr, yr)
domain = np.stack(XY, -1).reshape(-1, 2).T

plot_funcs = [
    [branin_hoo_fn, "Electric"],
    [branin_hoo_factory(*init_state), "Viridis"],  # default colorscale
    [branin_hoo_factory(*chain[-1].numpy()), "Blues"],
]
surfaces = [
    go.Surface(
        x=xr, y=yr, z=fn(domain).reshape(len(yr), -1), colorscale=cs, showscale=False
    )
    for fn, cs in plot_funcs
]
samples_plot = go.Scatter3d(x=xy[0], y=xy[1], z=z_true, mode="markers")
fig = go.Figure(data=[*surfaces, samples_plot])
title = "Branin-Hoo (bottom), initial surface (top), HMC final surface (middle)"
fig.update_layout(height=700, title_text=title)


# %%
fig = plt.figure(figsize=[12, 8])
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*xy, z_true, s=100, alpha=1, c="k")
ax.plot_surface(*XY, branin_hoo_fn(XY), alpha=0.7, cmap=cm.Greens)
ax.plot_surface(*XY, branin_hoo_factory(*bh_params_2x)(XY), alpha=0.7, cmap=cm.Blues)
ax.plot_surface(*XY, branin_hoo_factory(*chain[-1])(XY), alpha=0.7, cmap=cm.viridis)

plt.savefig("branin-hoo-hmc-test.pdf", bbox_inches="tight")
