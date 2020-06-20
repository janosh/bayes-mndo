# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import os

import json

from turbo import Turbo1
from tqdm import tqdm
from data import load_data, prepare_data
from objective import penalty, penalty_parallel

from chemhelp import mndo, units

mols_atoms, mols_coords, _, _, reference = load_data(query_size=1000, offset=0)
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

# with open("../parameters/parameters-pm3.json") as f:
#     # with open("../parameters/parameters-mndo-mean.json") as f:
#     start_params = json.loads(f.read())

with open("../parameters/parameters-opt-turbo.json", "r") as f:
    # with open("../parameters/parameters-mndo-mean.json", "r") as f:
    raw_json = f.read()
    mean_params = json.loads(raw_json)

with open("../parameters/parameters-mndo-std.json", "r") as f:
    raw_json = f.read()
    scale_params = json.loads(raw_json)

param_keys, _ = prepare_data(mols_atoms, mean_params)

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "n_procs": 2,
    "binary": "/home/reag2/PhD/second-year/bayes-mndo/mndo/mndo99_binary",
}


class MNDO:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.dim = len(kwargs["param_keys"])
        # self.lb = -3 * np.ones(self.dim)
        # self.ub = 3 * np.ones(self.dim)
        self.lb = -1 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)
        assert isinstance(kwargs["n_procs"], int) and kwargs["n_procs"] > 0
        self.n_procs = kwargs["n_procs"]

    def __call__(self, param_joblist):
        assert len(param_joblist[0]) == self.dim
        assert param_joblist[0].ndim == 1
        assert np.all([np.all(param_list <= self.ub) for param_list in param_joblist])
        assert np.all([np.all(param_list >= self.lb) for param_list in param_joblist])
        if self.n_procs > 1:
            fX = penalty_parallel(param_joblist, **self.kwargs)
        else:
            fX = []
            for param_list in tqdm(param_joblist):
                fX.append([penalty(param_list, **self.kwargs)])
            fX = np.array(fX)
        return fX


f = MNDO(kwargs)

# %%
# turbo = TurboM(
#     f=f,  # Handle to objective function
#     lb=f.lb,  # Numpy array specifying lower bounds
#     ub=f.ub,  # Numpy array specifying upper bounds
#     n_init=10,  # Number of initial bounds from an Symmetric Latin hypercube design
#     max_evals=1000,  # Maximum number of evaluations
#     n_trust_regions=5,  # Number of trust regions
#     batch_size=10,  # How large batch size TuRBO uses
#     verbose=True,  # Print information from each batch
#     use_ard=True,  # Set to true if you want to use ARD for the GP kernel
#     max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
#     n_training_steps=50,  # Number of steps of ADAM to learn the hypers
#     min_cuda=1024,  # Run on the CPU for small datasets
#     device="cpu",  # "cpu" or "cuda"
#     dtype="float64",  # float64 or float32
# )

turbo = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals=100,  # Maximum number of evaluations
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
turbo.optimize()

# %%
X = turbo.X  # Evaluated points
# NOTE we should hack the code so that we can see which trust regions each
# point came from this would allow for the figure to be coloured by region
fX = turbo.fX  # Observed values

param_names = ["-".join(tup) for tup in param_keys]
samples = np.hstack((fX, X))
df = pd.DataFrame(samples, columns=["penalty"] + param_names)
df = df.sort_values(by=["penalty"])
df.to_csv("turbo-samples.csv")

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

end_params = {atom_type: {} for atom_type, _ in param_keys}
for (atom_type, prop), param in zip(param_keys, x_best):
    end_params[atom_type][prop] = param

for atomtype in end_params:
    p, s, d = end_params[atomtype], scale_params[atomtype], mean_params[atomtype]
    for key in p:
        end_params[atomtype][key] = p[key] * s[key] + d[key]

with open("../parameters/parameters-opt-turbo-temp.json", "w") as f:
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
