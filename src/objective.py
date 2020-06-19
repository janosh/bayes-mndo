import numpy as np
from tqdm import trange

import pipelines
from chemhelp import mndo, units


def calc_err(props_list, ref_props=None, alpha=0.1, **kwargs):
    """
    Input:
        props_list: list of dictionaries of properties for each molecule
        ref_props: the target properties
    """
    calc_props = np.array([props["energy"] for props in props_list])

    # Change the units from electron volt to kcal/mol
    calc_props *= units.ev_to_kcalmol

    diff = ref_props - calc_props

    err = np.sqrt(np.nanmean((diff ** 2)))
    # Penalise the loss surface according to the number of non-converged
    # calculations. Currently we have a relatively naive choice of penalty.
    # it might be possible to design a smarter scheme
    n_failed = np.isnan(diff).sum()
    # reg = 13 * 666.0 * n_failed
    # reg = 13 * n_failed

    penalty = err * (1 + (alpha * n_failed ** 2))

    print(f"Penalty: {penalty} (Error: {err} + Failed: {n_failed})")

    return penalty


def penalty(
    param_list, param_keys, mean_params, scale_params, binary, filename, **kwargs
):
    """
    Input:
        param_list: array of params for different atoms
        param_keys: list of (atom_type, key) tuples for param_list
        ref_energies: np.array of ground truth atomic energies
        filename: file containing list of molecules for mndo calculation
    """

    mndo_options = {}
    if "scr" in kwargs:
        mndo_options["scr"] = kwargs["scr"]

    pipelines.set_params(
        param_list, param_keys, mean_params, scale_params, **mndo_options
    )

    props_list = pipelines.calculate(binary, filename, **mndo_options)

    error = calc_err(props_list, **kwargs)

    return error


def jacobian(param_list, dh, **kwargs):
    """
    Input:
        param_list: array of params for different atoms
        dh: small value for numerical gradients

    NOTE as jacobian parallel works with 1 proc do we need jacobian?
    """
    param_list = list(param_list)
    grad = np.zeros_like(param_list)

    for i in trange(len(param_list)):
        param_list[i] += dh
        forward = penalty(param_list, **kwargs)

        param_list[i] -= 2 * dh
        backward = penalty(param_list, **kwargs)

        de = forward - backward
        grad[i] = de / (2 * dh)

        param_list[i] += dh  # undo in-place changes to params for next iteration

    return grad


def penalty_parallel(params_joblist, n_procs=2, **kwargs):
    """
    Input:
        param_list: array of params for different atoms
        param_keys: list of (atom_type, key) tuples for param_list
        ref_energies: np.array of ground truth atomic energies
        filename: file containing list of molecules for mndo calculation
    """
    # maximum number of processes should be number of samples per batch
    n_procs = min(n_procs, len(params_joblist))

    props_lists = pipelines.calculate_parallel(
        params_joblist, n_procs=n_procs, **kwargs
    )

    penalty = np.array([[calc_err(props_list, **kwargs)] for props_list in props_lists])

    return penalty


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

    params_joblist = []
    dhs = np.zeros_like(param_list)
    for idx in range(len(param_list)):
        dhs[idx] = dh
        # forward
        params_joblist.append(param_list + dhs)
        # backward
        params_joblist.append(param_list - dhs)
        # reset dhs for next iter
        dhs[idx] = 0

    results = pipelines.calculate_parallel(params_joblist, n_procs=n_procs, **kwargs)

    grad = np.zeros_like(param_list)

    for i in range(len(param_list)):
        forward_props = results[2 * i]
        backward_props = results[2 * i + 1]

        penalty_forward = calc_err(forward_props, **kwargs)
        penalty_backward = calc_err(backward_props, **kwargs)

        de = penalty_forward - penalty_backward
        grad[i] = de / (2 * dh)

    return grad
