# %%
import argparse
import json
import os
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chemhelp import mndo, units
from tqdm import tqdm
from turbo import Turbo1, TurboM

from data import load_data, prepare_params
from objective import penalty, penalty_parallel

parser = argparse.ArgumentParser(description=("turbo optim"))

# data inputs
parser.add_argument(
    "--query",
    type=int,
    default=1000,
    metavar="INT",
    help="Number of input molecules",
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
    "--n-trust",
    "--nt",
    type=int,
    default=5,
    metavar="INT",
    help="Number of trust regions for optimisation",
)
parser.add_argument(
    "--max-evals",
    type=int,
    default=1000,
    metavar="INT",
    help="Maximum number of function evaluations",
)
parser.add_argument(
    "--batch-size",
    "--nb",
    type=int,
    default=10,
    metavar="INT",
    help="Number of function evaluations per iteration",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="plot the samples",
)

args = parser.parse_args(sys.argv[1:])

mols_atoms, mols_coords, _, _, reference = load_data(
    query_size=args.query, offset=args.offset
)
ref_energies = reference["binding_energy"].values

# Switch from Hartree to KCal/Mol
ref_energies *= units.hartree_to_kcalmol

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

root = os.path.abspath(__file__).split("/src", 1)[0]

with open(root + "/parameters/parameters-mndo-mean.json") as f:
    mean_params = json.loads(f.read())

with open(root + "/parameters/parameters-mndo-std.json") as f:
    scale_params = json.loads(f.read())

param_keys, _ = prepare_params(mols_atoms, mean_params)

kwargs = {
    "param_keys": param_keys,
    "filename": filename,
    "ref_props": ref_energies,
    "mean_params": mean_params,
    "scale_params": scale_params,
    "n_procs": n_procs,
    "binary": root + "/mndo/mndo99_binary",
    "scr": scrdir,
}


class MNDO:
    def __init__(self, **kwargs):
        assert kwargs["n_procs"] > 0, f"n_procs must be > 0, got {kwargs['n_procs']}"
        self.kwargs = kwargs
        self.n_procs = kwargs["n_procs"]
        self.dim = len(kwargs["param_keys"])
        self.lb = -1 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)

    def __call__(self, param_joblist):
        assert len(param_joblist[0]) == self.dim, "length mismatch"
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


objective_fn = MNDO(**kwargs)

if args.n_trust > 1:
    turbo = TurboM(
        f=objective_fn,  # Handle to objective function
        lb=objective_fn.lb,  # numpy array specifying lower bounds
        ub=objective_fn.ub,  # numpy array specifying upper bounds
        n_init=args.batch_size,  # initial bounds count from Symmetric Latin hypercube design
        max_evals=args.max_evals,  # Maximum number of evaluations
        n_trust_regions=args.n_trust,  # Number of trust regions
        batch_size=args.batch_size,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
else:
    turbo = Turbo1(
        f=objective_fn,  # Handle to objective function
        lb=objective_fn.lb,  # numpy array specifying lower bounds
        ub=objective_fn.ub,  # numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals=args.max_evals,  # Maximum number of evaluations
        batch_size=args.batch_size,  # How large batch size TuRBO uses
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
# NOTE need to scale these samples at some point
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

with open("parameters/parameters-opt-turbo.json", "w") as f:
    json.dump(end_params, f)


# %%
if args.plot:
    fig = plt.figure(figsize=(7, 5))
    matplotlib.rcParams.update({"font.size": 16})
    plt.plot(fX, "b.", ms=10)  # Plot all evaluated points as blue dots
    plt.plot(
        np.minimum.accumulate(fX), "r", lw=3
    )  # Plot cumulative minimum as a red line
    plt.xlim([0, len(fX)])
    plt.title("Average Energy Deviation")

    plt.tight_layout()
    plt.show()
