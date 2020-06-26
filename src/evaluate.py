import json
import os
import pathlib

import numpy as np

from chemhelp import mndo, units
from data import load_data, prepare_params
from objective import penalty

mols_atoms, mols_coords, _, _, reference = load_data(query_size=5000, offset=210)
ref_energies = reference["binding_energy"].values

# Switch from Hartree to KCal/Mol
ref_energies *= units.hartree_to_kcalmol

dh = 1e-5
n_procs = 2
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

root = os.path.abspath(__file__).split("/src", 1)[0]

with open(root + "/parameters/parameters-opt-turbo.json", "r") as f:
    # with open(root + "/parameters/parameters-opt-turbo-long.json", "r") as f:
    # with open(root + "/parameters/parameters-mndo-mean.json") as f:
    test_params = json.loads(f.read())

param_keys, test_params = prepare_params(mols_atoms, test_params)

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": None,
    "scale_params": None,
    "binary": root + "/mndo/mndo99_binary",
    "scr": scrdir,
}


rmse = penalty(test_params, **kwargs)  # objective function

print(f"RMSE: {rmse}")
print(ref_energies.mean())
print(f"Dummy: {np.mean(ref_energies-ref_energies.mean())}")
