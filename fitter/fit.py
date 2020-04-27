import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from scipy.optimize import minimize
import rmsd

import mndo

cachedir = ".pycache"
memory = joblib.Memory(cachedir, verbose=0)


@memory.cache
def load_data(data_file="data/qm9-reference.csv", offset=110, query_size=100):
    """
    Inputs:
        data_file (str): The data_file
        offset (int): The row offset for the data query
        query_size (int): The number of rows to return

    Returns:
        atom_list: List of chemical species for each molecule in query
        coords_list: List of species coordinates for each molecule in query
        charges: List of species charges for each molecule in query
        titles: List of names for each
        reference
    """

    reference = pd.read_csv(data_file)

    filenames = reference["name"]
    # energies = reference["binding energy"]

    atoms_list = []
    coord_list = []
    charges = []
    titles = []

    for filename in filenames:

        titles.append(filename)
        charges.append(0)

        filename = f"data/xyz/{filename}.xyz"
        atoms, coord = rmsd.get_coordinates_xyz(filename)

        atoms_list.append(atoms)
        coord_list.append(coord)

    atoms_list = atoms_list[offset : offset + query_size]
    coord_list = coord_list[offset : offset + query_size]
    charges = charges[offset : offset + query_size]
    titles = titles[offset : offset + query_size]
    reference = reference[offset : offset + query_size]

    return atoms_list, coord_list, charges, titles, reference


ignore_keys = [
    "DD2",
    "DD3",
    "PO1",
    "PO2",
    "PO3",
    "PO9",
    "HYF",
    "CORE",
    "EISOL",
    "FN1",
    "FN2",
    "FN3",
    "GSCAL",
    "BETAS",
    "ZS",
]


def minimize_params_scipy(
    mols_atoms,
    mols_coords,
    ref_energies,
    start_parameters,
    n_procs=1,
    method="PM3",
    ignore_keys=ignore_keys,
):
    """
    """
    filename = "_tmp_optimizer"
    txt = mndo.get_inputs(
        mols_atoms,
        mols_coords,
        np.zeros_like(mols_atoms),
        range(len(mols_atoms)),
        method=method,
    )

    with open(filename, "w") as f:
        f.write(txt)

    # find the species of atoms present in the batch of data
    atoms = [np.unique(mol_atoms) for mol_atoms in mols_atoms]
    atoms = list(itertools.chain(*atoms))
    atoms = np.unique(atoms)

    param_values = []
    param_keys = []
    parameters = {}

    # construct a dictionary of the parameters to optimise for each species
    for atom in atoms:
        atom_params = start_parameters[atom]

        current = {}

        for key in atom_params:

            if key in ignore_keys:
                continue

            value = atom_params[key]
            current[key] = value
            param_values.append(value)
            param_keys.append([atom, key])

        parameters[atom] = current

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

        calc_energies = np.array([p["energy"] for p in props_list])

        diff = ref_energies - calc_energies
        idxs = np.argwhere(np.isnan(diff))
        diff[idxs] = 700

        mae = np.abs(diff).mean()

        # print(f"penalty: {mae:.4g}")

        return mae

    def jacobian(params, dh=1e-6, debug=True):
        """
        Input:
        """

        # TODO Parallelt

        grad = np.zeros_like(params)

        for i, _ in enumerate(params):
            params[i] += dh
            forward = penalty(params)

            params[i] -= 2 * dh
            backward = penalty(params)

            de = forward - backward
            grad[i] = de / (2 * dh)

            params[i] += dh  # undo in-place changes to params for next iteration

        if debug:
            nm = np.linalg.norm(grad)
            print(f"penalty grad: {nm:.4g}")

        return grad

    res = minimize(
        penalty,  # objective function
        param_values,  # initial condition
        method="L-BFGS-B",
        jac=jacobian,
        options={"maxiter": 1000, "disp": True},
    )

    param_values = res.x
    error = penalty(param_values)

    # update parameters dictionary with values from minimizer
    for param, key in zip(param_values, param_keys):
        parameters[key[0]][key[1]] = param

    return parameters, error


def learning_curve(mols_atoms, mols_coords, reference_properties, start_parameters):
    """
    cross validate the parameters
    """

    five_fold = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

    # score = []

    for train_idxs, _ in five_fold.split(range(len(mols_atoms))):

        train_atoms = [mols_atoms[i] for i in train_idxs]
        train_coords = [mols_coords[i] for i in train_idxs]
        train_properties = reference_properties[train_idxs]

        # test_atoms = [mols_atoms[i] for i in test_idxs]
        # test_coords = [mols_coords[i] for i in test_idxs]
        # test_properties = reference_properties[test_idxs]

        train_parameters, _ = minimize_params_scipy(
            train_atoms, train_coords, train_properties, start_parameters
        )

        print(train_parameters)


def main():
    """
    """

    print("collect data")

    mols_atoms, mols_coords, _, _, reference = load_data()
    ref_energies = reference.iloc[:, 1].tolist()
    ref_energies = np.array(ref_energies)

    print("collect params")

    with open(args.parameters) as f:
        start_params = f.read()
        start_params = json.loads(start_params)

    end_params, error = minimize_params_scipy(
        mols_atoms, mols_coords, ref_energies, start_params
    )
    # end_params = learning_curve(mols_atoms, mols_coords, ref_energies, start_params)

    print(end_params)
    print(error)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="%(prog)s [options]")

    parser.add_argument("-f", "--format", action="store", help="", metavar="fmt")
    parser.add_argument("-s", "--settings", action="store", help="", metavar="json")
    parser.add_argument("-p", "--parameters", action="store", help="", metavar="json")
    parser.add_argument(
        "-o", "--results_parameters", action="store", help="", metavar="json"
    )
    parser.add_argument("-m --methods", action="store", help="", metavar="str")

    args = parser.parse_args()

    main()
    pass
