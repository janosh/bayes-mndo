import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import truncnorm
from tqdm import trange

import mndo
from data import load_data, prepare_data


def minimize_params_scipy(
    mols_atoms,
    mols_coords,
    ref_energies,
    n_procs=1,
    method="PM3",
):
    """
    """
    filename = "_tmp_optimizer"
    mndo.write_tmp_optimizer(mols_atoms, mols_coords, method)
    inputtxt = mndo.get_tmp_optimizer(mols_atoms, mols_coords, method)

    with open("../parameters/parameters-pm3-opt.json") as file:
        start_params = json.loads(file.read())

    param_keys, param_values = prepare_data(mols_atoms, start_params)
    # param_values = [np.random.normal() for _ in param_keys]
    param_values = [np.array(0.) for _ in param_keys]

    def penalty_properties(props_list):
        """
        """
        calc_energies = np.array([props["energy"] for props in props_list])
        diff = ref_energies - calc_energies
        idxs = np.argwhere(np.isnan(diff))
        diff[idxs] = 700.0

        error = (diff ** 2).mean()
        # error = np.abs(diff).mean()

        return error


    def penalty(param_list):
        """
        params: dict of params for different atoms
        ref_energies: np.array of ground truth atomic energies
        """
        # mndo expects params to be a dict, constructing that here
        # because scopt.minimze requires param_list to be a list
        params = {key[0]: {} for key in param_keys}
        for key, param in zip(param_keys, param_list):
            atom_type, prop = key
            params[atom_type][prop] = param

        mndo.set_params(params)

        props_list = mndo.calculate(filename)

        return penalty_properties(props_list)


    def jacobian(param_list, dh=1e-5, debug=True):
        """
        Input:
        """

        grad = np.zeros_like(param_list)

        for i in trange(len(param_list)):
            param_list[i] += dh
            forward = penalty(param_list)

            param_list[i] -= 2 * dh
            backward = penalty(param_list)

            de = forward - backward
            grad[i] = de / (2 * dh)

            param_list[i] += dh  # undo in-place changes to params for next iteration

        if debug:
            nm = np.linalg.norm(grad)
            print(f"penalty grad: {nm:.4g}")

        return grad


    def jacobian_parallel(param_list, dh=1e-5, procs=1):
        """
        """

        params_grad = mndo.numerical_jacobian(inputtxt, param_list, 
                                param_keys, n_procs=procs, dh=dh)

        grad = np.zeros_like(param_keys)

        for i, (atom, key) in enumerate(param_keys):
            forward_mols, backward_mols = params_grad[atom][key]

            penalty_forward = penalty_properties(forward_mols)
            penalty_backward = penalty_properties(backward_mols)

            de = penalty_forward - penalty_backward
            grad[i] = de/(2.0 * dh)

        return grad

    res = minimize(
        penalty,  # objective function
        param_values,  # initial condition
        method="L-BFGS-B",
        jac=jacobian_parallel,
        options={"maxiter": 1000, "disp": True},
    )

    param_values = res.x
    error = penalty(param_values)

    end_params = {key[0]: {} for key in param_keys}
    for key, param in zip(param_keys, param_values):
        atom_type, prop = key
        end_params[atom_type][prop] = param

    return end_params


def main():
    """
    """

    mols_atoms, mols_coords, _, _, reference = load_data()
    ref_energies = reference.iloc[:, 1].tolist()
    ref_energies = np.array(ref_energies)

    end_params, error = minimize_params_scipy(mols_atoms, mols_coords, ref_energies,)

    print(end_params)
    print(error)


if __name__ == "__main__":
    main()
    pass
