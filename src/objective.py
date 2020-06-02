import numpy as np
from tqdm import trange

import mndo


def calc_err(props_list, ref_props=None, **kwargs):
    """
    Input:
        props_list: list of dictionaries of properties for each molecule
        ref_props: the target properties
    """
    calc_props = np.array([props["energy"] for props in props_list])
    diff = ref_props - calc_props

    # ASK: What does this 700.0 do?
    idxs = np.argwhere(np.isnan(diff))
    diff[idxs] = 700.0

    err = (diff ** 2).mean()
    # err = np.abs(diff).mean()

    return err


def penalty(param_list, param_keys=None, filename=None, **kwargs):
    """
    Input:
        param_list: array of params for different atoms
        param_keys: list of (atom_type, key) tuples for param_list
        ref_energies: np.array of ground truth atomic energies
        filename: file containing list of molecules for mndo calculation
    """
    mean_params = kwargs["mean_params"]
    scale_params = kwargs["scale_params"]
    binary = kwargs["binary"]

    mndo.set_params(param_list, param_keys, mean_params, scale_params)

    props_list = mndo.calculate(binary, filename)

    return calc_err(props_list, **kwargs)


def jacobian(param_list, **kwargs):
    """
    Input:
        param_list: array of params for different atoms
        dh: small value for numerical gradients
    """
    param_list = list(param_list)
    # grad = np.zeros_like(param_list)
    grad = [0] * len(param_list)
    dh = kwargs.get("dh", 1e-5)

    for i in trange(len(param_list)):
        param_list[i] += dh
        forward = penalty(param_list, **kwargs)

        param_list[i] -= 2 * dh
        backward = penalty(param_list, **kwargs)

        de = forward - backward
        grad[i] = de / (2 * dh)

        param_list[i] += dh  # undo in-place changes to params for next iteration

    return grad


def jacobian_parallel(param_list, dh=1e-5, n_procs=2, **kwargs):
    """
    Input:
        param_list: array of params for different atoms
        param_keys: list of (atom_type, key) tuples for param_list
        mndo_input: str of input for the mndo calculations
        dh: small value for numerical gradients
        n_procs: number of cores to split computation over
    """
    # maximum number of processes should be one per parameter
    n_procs = min(n_procs, 2 * len(param_list))

    results = mndo.numerical_jacobian(param_list, n_procs=n_procs, dh=dh, **kwargs)

    # grad = np.zeros_like(param_list)
    grad = [0] * len(param_list)

    for i in range(len(param_list)):

        forward_props, backward_props = results[2 * i : 2 * i + 2]

        penalty_forward = calc_err(forward_props, **kwargs)
        penalty_backward = calc_err(backward_props, **kwargs)

        de = penalty_forward - penalty_backward
        grad[i] = de / (2 * dh)

    return grad
