# %%
import argparse
import json
import os
import pathlib
import sys
from functools import partial

import numpy as np
from scipy.optimize import minimize

from chemhelp import mndo, units
from data import load_data, prepare_params
from objective import jacobian_parallel, penalty

# %%
parser = argparse.ArgumentParser(description=("cgcnn"))

# data inputs
parser.add_argument(
    "--query", type=int, default=1000, metavar="INT", help="Number of input molecules",
)
parser.add_argument(
    "--offset",
    type=int,
    default=0,
    metavar="INT",
    help="Offset for selecting input molecules",
)
parser.add_argument(
    "--n-procs",
    "--np",
    type=int,
    default=2,
    metavar="INT",
    help="Number of processes used for parallelisation",
)
parser.add_argument(
    "--dh",
    type=float,
    default=1e-5,
    metavar="FLOAT",
    help="Size of perturbation for numerical gradients",
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=100,
    metavar="INT",
    help="Maximum number of optimizer iterations",
)

args = parser.parse_args(sys.argv[1:])

mols_atoms, mols_coords, _, _, reference = load_data(
    query_size=args.query, offset=args.offset
)
ref_energies = reference["binding_energy"].values

# Switch from Hartree to KCal/Mol
ref_energies *= units.hartree_to_kcalmol

dh = args.dh
n_procs = args.n_procs
method = "MNDO"

# NOTE we probably can refactor to remove the duplication of input files
filename = "_tmp_molecules"
scrdir = "_tmp_optim"

pathlib.Path(scrdir).mkdir(parents=True, exist_ok=True)

# TODO JCK At some point we need to evaluate non-zero molecules
n_molecules = len(mols_atoms)
mols_charges = np.zeros(n_molecules)
mols_names = np.arange(n_molecules)

mndo.write_input_file(
    mols_atoms,
    mols_coords,
    mols_charges,
    mols_names,
    method,
    os.path.join(scrdir, filename),
    read_params=True,
)

# %%
root = os.path.abspath(__file__).split("/src", 1)[0]


with open(root + "/parameters/parameters-mndo-mean.json", "r") as f:
    mean_params = json.loads(f.read())

with open(root + "/parameters/parameters-mndo-std.json", "r") as f:
    scale_params = json.loads(f.read())

param_keys, _ = prepare_params(mols_atoms, mean_params)
param_values = [np.random.normal() for _ in param_keys]
# param_values = [0.0 for _ in param_keys]

ps = [param_values]


kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "n_procs": n_procs,
    "dh": dh,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "binary": root + "/mndo/mndo99_binary",
    "scr": scrdir,
}

# %%
res = minimize(
    partial(penalty, **kwargs),  # objective function
    param_values,  # initial condition
    method="L-BFGS-B",
    # jac=partial(jacobian, **kwargs),
    jac=partial(jacobian_parallel, **kwargs),
    options={"maxiter": args.max_iter, "disp": True},
    callback=lambda p: ps.append(p),  # captures intermediate states of optimization
)
param_values = res.x


# %%
end_params = {atom_type: {} for atom_type, _ in param_keys}
for (atom_type, prop), param in zip(param_keys, param_values):
    end_params[atom_type][prop] = param

for atomtype in end_params:
    p, s, d = end_params[atomtype], scale_params[atomtype], mean_params[atomtype]
    for key in p:
        end_params[atomtype][key] = p[key] * s[key] + d[key]

with open(root + "/parameters/parameters-opt-scipy.json", "w") as f:
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
