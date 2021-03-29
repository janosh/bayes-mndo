import copy
import glob
import json
import os
from pathlib import Path

import pytest
import rmsd

from src.chemhelp import cheminfo, misc, mndo, units

SCRDIR = "_tmp_test"


def clean_scratch(dirname):

    for f in glob.glob(dirname + "/fort.*"):
        os.remove(f)

    return


def setup_multi_xyz():

    atoms_list = []
    coord_list = []
    charg_list = []

    filenames = range(1, 8)
    filenames = [f"{x:06d}" for x in filenames]
    filenames = [f"dsgdb9nsd_{x}" for x in filenames]

    XYZ_DIRNAME = "data/xyz/"

    for filename in filenames:

        filename_xyz = os.path.join(XYZ_DIRNAME, filename + ".xyz")
        atoms, coord = rmsd.get_coordinates_xyz(filename_xyz)

        atoms_list.append(atoms)
        coord_list.append(coord)
        charg_list.append(0)

    return atoms_list, coord_list, charg_list, filenames


def test_set_param():

    scrdir = "_tmp_test"
    filename = "_tmp_input"

    Path(scrdir).mkdir(parents=True, exist_ok=True)

    parameters = dict()
    parameters["O"] = dict()
    parameters["O"]["USS"] = 666.0

    # Set parameters
    mndo.set_params(parameters, scr=scrdir)

    atoms = ["O", "N", "C", "N", "N", "H"]

    coords = [
        [-0.0593325887, 1.2684201211, 0.0095178503],
        [1.1946293712, 1.771776509, 0.0001229152],
        [1.9590217387, 0.7210517427, -0.0128069641],
        [1.2270421979, -0.4479406483, -0.0121559722],
        [0.0119302176, -0.1246338293, 0.0012973737],
        [3.0355546734, 0.7552313348, -0.0229864829],
    ]

    # Create mndo input format
    inptxt = mndo.get_input(atoms, coords, 0, title="title", read_params=True)

    with open(str(Path(scrdir).joinpath(filename)), "w") as f:
        f.write(inptxt)

    calculations = mndo.run_mndo_file(filename, scr=scrdir)
    calculations = list(calculations)
    lines = calculations[0]
    idx = misc.get_index(lines, "USS")
    line = lines[idx]
    line = line.split()

    value = float(line[-1])

    assert value == 666.0

    return


def test_water():

    scrdir = "_tmp_test"
    filename = "_tmp_water"

    Path(scrdir).mkdir(parents=True, exist_ok=True)

    # Get molecule
    molobj = cheminfo.generate_conformers("O", max_conf=1, min_conf=1)
    atoms, coord, charge = cheminfo.molobj_to_axyzc(molobj, atom_type=str)

    # Optimizer header
    header = """MNDO MULLIK PRECISE charge={charge} jprint=5
nextmol=-1
TITLE {title}"""

    # Set input file
    inptxt = mndo.get_input(
        atoms,
        coord,
        charge,
        title="water optimize example",
        read_params=False,
        header=header,
        optimize=True,
    )

    with open(str(Path(scrdir).joinpath(filename)), "w") as f:
        f.write(inptxt)

    # Run mndo
    calculations = mndo.run_mndo_file(filename, scr=scrdir)
    calculations = list(calculations)

    lines = calculations[0]
    properties = mndo.get_properties(lines)

    water_atomization = properties["energy"] * units.ev_to_kcalmol

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return


def test_params_error():

    scrdir = SCRDIR
    filename = "_tmp_multimol"
    method = "MNDO"

    options = {"mndo_cmd": "mndo"}  # set mndo path

    clean_scratch(scrdir)

    # Setup multiple molecules
    mols_atoms, mols_coords, mols_charges, mols_names = setup_multi_xyz()

    # Write multi input file
    mndo.write_input_file(
        mols_atoms,
        mols_coords,
        mols_charges,
        mols_names,
        method,
        os.path.join(scrdir, filename),
    )

    # Set parameters
    with open("parameters/parameters-mndo-mean.json", "r") as file:
        raw_json = file.read()
        mean_params = json.loads(raw_json)

    mndo.set_params(mean_params, scr=scrdir)

    # Calculate and collect the results
    results = mndo.calculate(filename, scr=scrdir, **options)

    for properties in results:

        assert properties is not None
        assert type(properties) is dict
        assert type(float(properties["energy"])) is float

    return


def test_params_parallel():

    # Test on multi core
    n_procs = 1

    scrdir = SCRDIR
    filename = "_tmp_multimol"
    method = "MNDO"

    # Prepare some parameters
    with open("parameters/parameters-mndo-mean.json", "r") as file:
        raw_json = file.read()
        mean_params = json.loads(raw_json)

    parameter_list = []
    for _ in range(200):
        parameter_list.append(copy.deepcopy(mean_params))

    # Clean
    clean_scratch(scrdir)

    # Set input file
    mols_atoms, mols_coords, mols_charges, mols_names = setup_multi_xyz()

    # Write multi input file
    mndo.write_input_file(
        mols_atoms,
        mols_coords,
        mols_charges,
        mols_names,
        method,
        os.path.join(scrdir, filename),
    )

    results = mndo.calculate_parameters(
        filename, parameter_list, scr=scrdir, n_procs=n_procs
    )

    for result in results:

        assert type(result) is list
        assert type(result[0]) is dict
        assert "energy" in result[0]

    return


def main():

    test_params_parallel()

    return


if __name__ == "__main__":
    main()
