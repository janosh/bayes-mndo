import mndo
import numpy as np


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

    print(f"mse: {mse:10.2f}")

    return mse
