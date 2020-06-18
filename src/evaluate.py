import json
import numpy as np

import mndo
from data import load_data, prepare_data
from objective import penalty


# NOTE choosing offest 0 puts C2H2 in training set which has
# the strange issue with not giving ionisation energy.
mols_atoms, mols_coords, _, _, reference = load_data(query_size=9000, offset=1000)
ref_energies = reference["binding_energy"].values

method = "MNDO"
filename = "_tmp_molecules"
mndo.write_tmp_optimizer(mols_atoms, mols_coords, filename, method)

with open("../parameters/parameters-opt-turbo.json", "r") as f:
    # with open("../parameters/parameters-opt-turbo-long.json", "r") as f:
    # with open("../parameters/parameters-mndo-mean.json") as f:
    test_params = json.loads(f.read())

# with open("../parameters/parameters-mndo-mean.json", "r") as f:
#     raw_json = f.read()
#     mean_params = json.loads(raw_json)

# with open("../parameters/parameters-mndo-std.json", "r") as f:
#     raw_json = f.read()
#     scale_params = json.loads(raw_json)

param_keys, test_params = prepare_data(mols_atoms, test_params)

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": None,
    "scale_params": None,
    "binary": "/home/reag2/PhD/second-year/bayes-mndo/mndo/mndo99_binary",
}


rmse = penalty(test_params, **kwargs)  # objective function

print(f"RMSE: {rmse}")
print(ref_energies.mean())
print(f"Dummy: {np.mean(ref_energies-ref_energies.mean())}")
