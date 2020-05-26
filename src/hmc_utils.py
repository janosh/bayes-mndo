import tensorflow as tf
import tensorflow_probability as tfp


# @tf.function
# @tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """Since this is bulk of the computation, using @tf.function
    here to compile a static graph for tfp.mcmc.sample_chain significantly improves
    performance, especially when enabling XLA (Accelerated Linear Algebra).
    https://tensorflow.org/xla#explicit_compilation_with_tffunction
    https://github.com/tensorflow/probability/issues/728#issuecomment-573704750
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)


def trace_fn_nuts(cs, kr, summary_writer=None, hist_freq=10, callbacks=[]):
    """
    cs: current state, tensor or list of tensors
    kr: kernel results
    hist_freq: record histograms every n steps
    callbacks: list of functions taking the current_state,
        output is added to trace
    """
    step = tf.cast(kr.step, tf.int64)
    nuts = kr.inner_results
    target_log_prob = nuts.target_log_prob

    with summary_writer.as_default():
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("log likelihood (mse)", target_log_prob)
        tf.summary.scalar("energy", nuts.energy)
        tf.summary.scalar("log accept ratio", nuts.log_accept_ratio)
        tf.summary.scalar("leapfrogs taken", nuts.leapfrogs_taken)
        with tf.summary.record_if(tf.equal(step % hist_freq, 0)):
            tf.summary.histogram("step size", nuts.step_size)

        # tf.summary.scalar("step size", kr.new_step_size)
        # tf.summary.scalar("decay rate", kr.decay_rate)
        # tf.summary.scalar("error sum", kr.error_sum)

    if callbacks:
        return target_log_prob, [cb(*cs) for cb in callbacks]
    return target_log_prob


def get_nuts_kernel(target_log_prob_fn, step_size, n_adapt_steps):
    kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size)
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=n_adapt_steps,
        # pkr: previous kernel results, ss: step size
        step_size_setter_fn=lambda pkr, new_ss: pkr._replace(step_size=new_ss),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    )
    return adaptive_kernel


def trace_fn_hmc(cs, kr, summary_writer=None, hist_freq=10, callbacks=[]):
    """
    cs: current state, tensor or list of tensors
    kr: kernel results
    hist_freq: record histograms every n steps
    callbacks: list of functions taking the current_state,
        output is added to trace
    """
    step = tf.cast(kr.step, tf.int64)
    hmc = kr.inner_results
    target_log_prob = hmc.accepted_results.target_log_prob

    with summary_writer.as_default():
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("log likelihood (mse)", target_log_prob)
        tf.summary.scalar("prop mse", hmc.proposed_results.target_log_prob)
        tf.summary.scalar("log accept ratio", hmc.log_accept_ratio)

    if callbacks:
        return target_log_prob, [cb(*cs) for cb in callbacks]
    return target_log_prob


def get_hmc_kernel(target_log_prob_fn, step_size, n_adapt_steps):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=20,
    )
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=n_adapt_steps,
        # pkr: previous kernel results, ss: step size
        # step_size_setter_fn=lambda pkr, new_ss: pkr._replace(step_size=new_ss),
        # step_size_getter_fn=lambda pkr: pkr.step_size,
        # log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    )
    return adaptive_kernel
