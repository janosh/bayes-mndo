import numpy as np
from tqdm import trange

import mndo


def penalty(param_vals, param_keys, ref_energies, filename):
    """
    params: dict of params for different atoms
    ref_energies: np.array of ground truth atomic energies
    """
    # mndo expects params to be a dict, constructing that here
    # because scopt.minimze requires param_list to be a list
    params = {key[0]: {} for key in param_keys}
    for key, param in zip(param_keys, param_vals):
        atom_type, prop = key
        params[atom_type][prop] = param

    mndo.set_params(params)
    preds = mndo.calculate(filename)
    pred_energies = np.array([p["energy"] for p in preds])

    diff = ref_energies - pred_energies

    mse = (diff ** 2).mean()

    # print(f"mse: {mse:10.2f}")

    return mse


def jacobian(*args, dh=1e-6):
    param_list, *rest = args

    grad = np.zeros_like(param_list)
    param_list = list(param_list)

    for i in trange(len(param_list)):
        param_list[i] += dh
        forward = penalty(param_list, *rest)

        param_list[i] -= 2 * dh
        backward = penalty(param_list, *rest)

        de = forward - backward
        grad[i] = de / (2 * dh)

        param_list[i] += dh  # undo in-place changes to params for next iteration

    norm = np.linalg.norm(grad)
    print(f"penalty grad: {norm:.4g}")

    return grad
