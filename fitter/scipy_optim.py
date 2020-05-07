import json
import time
from functools import partial

import numpy as np
from scipy.optimize import minimize
from tqdm import trange

import mndo
from data import load_data, prepare_data
from objective import jacobian, jacobian_parallel, penalty


def minimize_params_scipy(
    mols_atoms, mols_coords, ref_energies, dh=1e-5, n_procs=1, method="PM3",
):
    filename = "_tmp_optimizer"
    mndo.write_tmp_optimizer(mols_atoms, mols_coords, method)

    mndo_input = mndo.get_inputs(
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
    param_values = [0.0 for _ in param_keys]

    ps = [param_values]

    def reporter(p):
        """Reporter function to capture intermediate states of optimization."""
        ps.append(p)

    kwargs = {
        "param_keys": param_keys,
        "filename": filename,
        "mndo_input": mndo_input,
        "n_procs": n_procs,
        "dh": dh,
        "ref_props": ref_energies,
    }
    try:
        res = minimize(
            partial(penalty, **kwargs),  # objective function
            param_values,  # initial condition
            method="L-BFGS-B",
            # jac=partial(jacobian, **kwargs),
            jac=partial(jacobian_parallel, **kwargs),
            options={"maxiter": 1000, "disp": True},
            callback=reporter,
        )
        param_values = res.x
    except IndexError:
        param_values = ps[-1]
        pass
    except KeyboardInterrupt:
        param_values = ps[-1]
        pass

    end_params = {atom_type: {} for atom_type, _ in param_keys}
    for (atom_type, prop), param in zip(param_keys, param_values):
        end_params[atom_type][prop] = param

    return end_params


def main():
    mols_atoms, mols_coords, _, _, reference = load_data(query_size=20)
    ref_energies = reference.iloc[:, 1].tolist()
    ref_energies = np.array(ref_energies)

    end_params = minimize_params_scipy(mols_atoms, mols_coords, ref_energies,)

    with open("parameters-opt.json", "w") as fp:
        json.dump(end_params, fp)


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
