import json
import time

import numpy as np
from scipy.optimize import minimize
from tqdm import trange

import mndo
from data import load_data, prepare_data


def minimize_params_scipy(
    mols_atoms, mols_coords, ref_energies, n_procs=1, method="PM3",
):
    """
    """
    filename = "_tmp_optimizer"
    mndo.write_tmp_optimizer(mols_atoms, mols_coords, method)
    inputtxt = mndo.get_inputs(
        mols_atoms,
        mols_coords,
        np.zeros_like(mols_atoms),
        range(len(mols_atoms)),
        method,
    )

    with open("../parameters/parameters-pm3-opt.json") as file:
        start_params = json.loads(file.read())

    param_keys, param_values = prepare_data(mols_atoms, start_params)
    # param_values = [np.random.normal() for _ in param_keys]
    param_values = [np.array(0.0) for _ in param_keys]

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
        Input:
            param_list: array of params for different atoms
            param_keys: list of (atom_type, key) tuples for param_list
            ref_energies: np.array of ground truth atomic energies
        """
        mndo.set_params(param_list, param_keys)

        props_list = mndo.calculate(filename)

        return penalty_properties(props_list)

    def jacobian(param_list, dh=1e-5, debug=True):
        """
        Input:
            param_list: array of params for different atoms
            dh: small value for numerical gradients
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
        Input:
            param_list: array of params for different atoms
            dh: small value for numerical gradients
            procs: number of cores to split computation over
        """
        # maximum number of processes should be one per parameter
        procs = min(procs, len(param_keys))

        params_grad = mndo.numerical_jacobian(
            inputtxt, param_list, param_keys, n_procs=procs, dh=dh
        )

        grad = np.zeros_like(param_list)

        for i, (atom, key) in enumerate(param_keys):
            forward_mols, backward_mols = params_grad[atom][key]

            penalty_forward = penalty_properties(forward_mols)
            penalty_backward = penalty_properties(backward_mols)

            de = penalty_forward - penalty_backward
            grad[i] = de / (2.0 * dh)

        return grad

    res = minimize(
        penalty,  # objective function
        param_values,  # initial condition
        method="L-BFGS-B",
        # jac=jacobian,
        jac=jacobian_parallel,
        options={"maxiter": 1000, "disp": True},
    )

    param_values = res.x
    # error = penalty(param_values)

    end_params = {key[0]: {} for key in param_keys}
    for key, param in zip(param_keys, param_values):
        atom_type, prop = key
        end_params[atom_type][prop] = param

    return end_params


def main():
    """
    """

    mols_atoms, mols_coords, _, _, reference = load_data(query_size=200)
    ref_energies = reference.iloc[:, 1].tolist()
    ref_energies = np.array(ref_energies)

    end_params = minimize_params_scipy(mols_atoms, mols_coords, ref_energies,)

    with open('parameters-opt.json', 'w') as fp:
        json.dump(end_params, fp)

    print(end_params)



if __name__ == "__main__":
    main()
    pass


# timing code

# dh = 1e-5

# t = time.time()
# grad = jacobian(param_values, dh=dh)
# nm = np.linalg.norm(grad)
# secs = time.time() - t
# print("penalty grad: {:10.2f} time: {:10.2f}".format(nm, secs))

# t = time.time()
# grad = jacobian_parallel(param_values, procs=2, dh=dh)
# nm = np.linalg.norm(grad)
# secs = time.time() - t
# print("penalty grad: {:10.2f} time: {:10.2f}".format(nm, secs))

# exit()
