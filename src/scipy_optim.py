import json
from functools import partial

import numpy as np
from scipy.optimize import minimize

import mndo
from data import load_data, prepare_data
from objective import jacobian, jacobian_parallel, penalty


# NOTE choosing offest 0 puts C2H2 in training set which has
# the strange issue with not giving ionisation energy.
mols_atoms, mols_coords, _, _, ref_energies = load_data(query_size=100, offset=110)
ref_energies = reference["binding_energy"].values

end_params = minimize_params_scipy(mols_atoms, mols_coords, ref_energies, method="MNDO")

dh = 1e-5
n_procs = 2
method = "MNDO"

# NOTE we probably can refactor to remove the duplication of input files
filename = "_tmp_molecules"
mndo.write_tmp_optimizer(mols_atoms, mols_coords, filename, method)

# with open("../parameters/parameters-pm3.json") as file:
#     # with open("../parameters/parameters-mndo-mean.json") as file:
#     start_params = json.loads(file.read())

with open("../parameters/parameters-mndo-mean.json", "r") as file:
    raw_json = file.read()
    mean_params = json.loads(raw_json)

with open("../parameters/parameters-mndo-std.json", "r") as file:
    raw_json = file.read()
    scale_params = json.loads(raw_json)

# param_keys, param_values = prepare_data(mols_atoms, start_params)
param_keys, _ = prepare_data(mols_atoms, mean_params)
param_values = [np.random.normal() for _ in param_keys]
# param_values = [0.0 for _ in param_keys]

ps = [param_values]


def reporter(p):
    """Reporter function to capture intermediate states of optimization."""
    ps.append(p)


kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "n_procs": n_procs,
    "dh": dh,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "binary": "/home/reag2/PhD/second-year/bayes-mndo/mndo/mndo99_binary",
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

for atomtype in end_params:
    p, s, d = end_params[atomtype], scale_params[atomtype], mean_params[atomtype]
    for key in p:
        end_params[atomtype][key] = p[key] * s[key] + d[key]

with open("../parameters/parameters-opt-scipy.json", "w") as f:
    json.dump(end_params, f)


# # timing code

# t = time.time()
# grad = jacobian(param_values, **kwargs)
# nm = np.linalg.norm(grad)
# secs = time.time() - t
# print("penalty grad: {:10.2f} time: {:10.2f}".format(nm, secs))

# t = time.time()
# grad = jacobian_parallel(param_values, **kwargs)
# nm = np.linalg.norm(grad)
# secs = time.time() - t
# print("penalty grad: {:10.2f} time: {:10.2f}".format(nm, secs))

# exit()
