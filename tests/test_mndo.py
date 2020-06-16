
import pytest
from pathlib import Path

from context import src

from src.chemhelp import mndo
from src.chemhelp import misc
from src.chemhelp import cheminfo
from src.chemhelp import units


def test_set_param():

    scrdir = "_tmp_test"
    filename = "_tmp_input"

    Path(scrdir).mkdir(parents=True, exist_ok=True)

    parameters = dict()
    parameters["O"] = dict()
    parameters["O"]["USS"] = 666.0

    # Set parameters
    mndo.set_params(parameters, cwd=scrdir)

    atoms = ['O','N','C','N','N','H']

    coords = [
        [ -0.0593325887, 1.2684201211  , 0.0095178503  ],
        [ 1.1946293712 , 1.771776509   , 0.0001229152  ],
        [ 1.9590217387 , 0.7210517427  , -0.0128069641 ],
        [ 1.2270421979 , -0.4479406483 , -0.0121559722 ],
        [ 0.0119302176 , -0.1246338293 , 0.0012973737  ],
        [ 3.0355546734 , 0.7552313348  , -0.0229864829 ],
    ]

    # Create mndo input format
    inptxt = mndo.get_input(atoms, coords, 0, title="title", read_params=True)

    with open(str(Path(scrdir).joinpath(filename)), 'w') as f:
        f.write(inptxt)

    calculations = mndo.run_mndo_file(filename, cwd=scrdir)
    calculations = list(calculations)
    lines = calculations[0]
    properties = mndo.get_properties(lines)
    idx = misc.get_index(lines, "USS")
    line = lines[idx]
    line = line.split()

    value = float(line[-1])

    assert value == 666.0

    return


def test_water():

    smi = "O"

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
    inptxt = mndo.get_input(atoms, coord, charge,
        title="water optimize example",
        read_params=False,
        header=header,
        optimize=True)

    with open(str(Path(scrdir).joinpath(filename)), 'w') as f:
        f.write(inptxt)

    # Run mndo
    calculations = mndo.run_mndo_file(filename, cwd=scrdir)
    calculations = list(calculations)

    lines = calculations[0]
    properties = mndo.get_properties(lines)

    water_atomization = properties["energy"]*units.ev_to_kcalmol

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return


def main():

    return


if __name__ == "__main__":
    main()
