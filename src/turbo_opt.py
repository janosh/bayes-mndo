# %%
from turbo import TurboM
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import json
import mndo
from data import load_data, prepare_data
from objective import penalty

# %%
mols_atoms, mols_coords, _, _, reference = load_data(query_size=500)
ref_energies = reference.iloc[:, 1].tolist()
ref_energies = np.array(ref_energies)

method = "MNDO"
filename = "_tmp_optimizer"
mndo.write_tmp_optimizer(mols_atoms, mols_coords, filename, method)

mndo_input = mndo.get_inputs(
    mols_atoms, mols_coords, np.zeros_like(mols_atoms), range(len(mols_atoms)), method,
)

# with open("../parameters/parameters-pm3.json") as f:
#     # with open("../parameters/parameters-mndo-mean.json") as f:
#     start_params = json.loads(f.read())

# with open("../parameters/parameters-opt-turbo.json", "r") as f:
with open("../parameters/parameters-mndo-mean.json", "r") as f:
    # with open("../parameters/parameters-opt-scipy.json", "r") as f:
    raw_json = f.read()
    mean_params = json.loads(raw_json)

with open("../parameters/parameters-mndo-std.json", "r") as f:
    raw_json = f.read()
    scale_params = json.loads(raw_json)

# param_keys, param_values = prepare_data(mols_atoms, start_params)
param_keys, _ = prepare_data(mols_atoms, mean_params)
param_values = [np.random.normal() for _ in param_keys]
# param_values = [0.0 for _ in param_keys]

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "binary": "/home/reag2/PhD/second-year/bayes-mndo/mndo/mndo99_binary",
}


class MNDO:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.dim = len(kwargs["param_keys"])
        self.lb = -0.5 * np.ones(self.dim)
        self.ub = 0.5 * np.ones(self.dim)

    def __call__(self, param_list):
        assert len(param_list) == self.dim
        assert param_list.ndim == 1
        assert np.all(param_list <= self.ub) and np.all(param_list >= self.lb)

        return penalty(param_list, **self.kwargs)


f = MNDO(kwargs)

# %%
turbo_m = TurboM(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=10,  # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=1000,  # Maximum number of evaluations
    n_trust_regions=5,  # Number of trust regions
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

# %%

print("Optimise")
turbo_m.optimize()

# %%
X = turbo_m.X  # Evaluated points
# NOTE we should hack the code so that we can see which trust regions each point came from
# this would allow for the figure to be coloured by region
fX = turbo_m.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

end_params = {atom_type: {} for atom_type, _ in param_keys}
for (atom_type, prop), param in zip(param_keys, x_best):
    end_params[atom_type][prop] = param

for atomtype in end_params:
    p, s, d = end_params[atomtype], scale_params[atomtype], mean_params[atomtype]
    for key in p:
        end_params[atomtype][key] = p[key] * s[key] + d[key]

with open("../parameters/parameters-opt-turbo.json", "w") as f:
    json.dump(end_params, f)

# %%
fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({"font.size": 16})
plt.plot(fX, "b.", ms=10)  # Plot all evaluated points as blue dots
plt.plot(np.minimum.accumulate(fX), "r", lw=3)  # Plot cumulative minimum as a red line
plt.xlim([0, len(fX)])
plt.title("Average Energy Deviation")

plt.tight_layout()
plt.show()


# %%
