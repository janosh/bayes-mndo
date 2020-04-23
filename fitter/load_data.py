import pandas as pd
import rmsd
import itertools


def load_data(data_file="data/qm9-reference.csv", offset=110, query_size=10):
    """
    Inputs:
        data_file (str): The data_file
        offset (int): The row offset for the data query
        query_size (int): The number of rows to return

    Returns:
        atom_list: List of chemical species for each molecule in query
        coords_list: List of species coordinates for each molecule in query
        charges: List of species charges for each molecule in query
        filenames: List of names for each reference
    """
    reference = pd.read_csv(data_file)

    atoms_list, coord_list, charges = [], [], []
    filenames = reference["name"]

    for filename in filenames:
        filename = f"data/xyz/{filename}.xyz"
        atoms, coords = rmsd.get_coordinates_xyz(filename)

        charges.append(0)
        atoms_list.append(atoms)
        coord_list.append(coords)

    end = offset + query_size
    atoms_list = atoms_list[offset:end]
    coord_list = coord_list[offset:end]
    charges = charges[offset:end]
    filenames = filenames[offset:end]
    reference = reference[offset:end]

    return atoms_list, coord_list, charges, filenames, reference


# fmt: off
ignore_keys = ["DD2", "DD3", "PO1", "PO2", "PO3", "PO9", "HYF", "CORE", "EISOL", "FN1", "FN2", "FN3", "GSCAL", "BETAS", "ZS"]
# fmt: on


def prepare_data(mols_atoms, start_params, ignore_keys=ignore_keys):
    atoms = [np.unique(mol_atoms) for mol_atoms in mols_atoms]
    atoms = list(itertools.chain(*atoms))
    atoms = np.unique(atoms)

    param_keys, param_values = [], []

    # select atom params
    for atom in atoms:
        atom_params = start_params[atom]

        for key in atom_params:
            if key in ignore_keys:
                continue

            value = atom_params[key]
            param_keys.append([atom, key])
            param_values.append(value)

    return param_keys, param_values
