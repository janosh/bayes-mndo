# %%
from datetime import datetime
import numpy as np
from functools import partial

import tensorflow as tf

from hmc_utils import sample_chain, trace_fn, get_nuts_kernel
from bo_bench import (
    branin_hoo_params,
    sample_branin_hoo,
    branin_hoo_factory,
    branin_hoo_fn,
)
import plotly.graph_objects as go


# %%
# Plot the Branin-Hoo surface
xr = np.linspace(-5, 15, 21)
yr = np.linspace(0, 10, 11)
domain = np.stack(np.meshgrid(xr, yr), -1).reshape(-1, 2).T

surface = go.Surface(x=xr, y=yr, z=branin_hoo_fn(domain).reshape(len(yr), -1))
fig = go.Figure(data=[surface])
fig.update_layout(height=700, title_text="Branin-Hoo function")


# %%
# Generate random data set
xy, z_true = sample_branin_hoo(100)


def target_log_prob_fn(param_vals):
    z_pred = branin_hoo_factory(*param_vals)(xy)
    return -tf.metrics.mse(z_true, z_pred)


# %%
now = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_dir = f"runs/hmc-trace/{now}"
summary_writer = tf.summary.create_file_writer(log_dir)


# %%
# Casting step_size and init_state needed due to TFP bug
# https://github.com/tensorflow/probability/issues/904#issuecomment-624272845
step_size = tf.cast(1e-2, tf.float64)
init_state = [v * 1.5 for v in branin_hoo_params.values()]
n_adapt_steps = 20

chain, trace, final_kernel_results = sample_chain(
    num_results=40,
    current_state=tf.constant(init_state, tf.float64),
    kernel=get_nuts_kernel(target_log_prob_fn, step_size, n_adapt_steps),
    return_final_kernel_results=True,
    # trace_fn=partial(trace_fn, summary_writer=summary_writer),
)
burnin, samples = chain[:n_adapt_steps], chain[n_adapt_steps:]


# %%
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
