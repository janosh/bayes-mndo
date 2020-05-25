import numpy as np


def branin_hoo_factory(a, b, c, r, s, t):
    def branin_hoo(x):
        # f(x) = a(y - b*x^2 + c*x - r)^2 + s (1 - t) cos(x) + s
        return (
            a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
            + s * (1 - t) * np.cos(x[0])
            + s
        )

    return branin_hoo


branin_hoo_params = dict(
    a=1, b=5.1 / (4 * np.pi ** 2), c=5 / np.pi, r=6, s=10, t=1 / (8 * np.pi)
)


def branin_hoo_fn(x):
    """The Branin-Hoo function is a popular benchmark for Bayesian optimization.
    """
    z = branin_hoo_factory(**branin_hoo_params)(x)
    return z


def sample_branin_hoo(n_samples, domain=[[-5, 15], [0, 10]]):
    """Take samples from the Branin-Hoo function.

    Args:
        n_samples (int): number of samples to draw

    Returns:
        np.array: 2d array of x, y z points
        np.array: 1d array of z points
    """
    [x_min, x_max], [y_min, y_max] = domain

    xy = np.random.uniform(
        low=[x_min, y_min], high=[x_max, y_max], size=(n_samples, 2)
    ).T
    z = branin_hoo_fn(xy)

    return xy, z
