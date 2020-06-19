import json
import multiprocessing as mp
import os
import shutil
import subprocess
from functools import partial

import numpy as np
from tqdm import tqdm


# DEPRECIATED!!!


def get_index(lines, pattern):
    for i, line in enumerate(lines):
        if pattern in line:
            return i
    return None


def get_indices(lines, pattern, stop_pattern=None):

    idxs = []

    for i, line in enumerate(lines):
        if pattern in line:
            idxs.append(i)
            continue

        if stop_pattern and stop_pattern in line:
            break

    return idxs


def get_indexes_patterns(lines, patterns):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None] * n_patterns

    for i, line in enumerate(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs


def get_rev_indices(lines, patterns):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None] * n_patterns

    for i, line in reversed(list(enumerate(lines))):
        for ip in i_patterns:
            pattern = patterns[ip]
            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs


def execute(cmd, filename, cwd):
    """
    Call the MNDO fortran binary. This function requires `mndo` to be in path.
    """

    if cwd is not None:
        job_cmd = f"cd {cwd}; {cmd} < {filename}"
    else:
        job_cmd = f"{cmd} < {filename}"

    popen = subprocess.Popen(
        job_cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True,
    )

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_mndo_file(binary, filename, cwd=None):
    """
    Runs mndo on the given input file. Yields lists of lines for each
    molecule as the program completes.
    """
    cmd = os.path.expanduser(binary)
    lines = execute(cmd, filename, cwd)

    molecule_lines = []

    # Lines is an iterator object
    for line in lines:

        line = line.strip()
        molecule_lines.append(line.strip("\n"))

        if "STATISTICS FOR RUNS WITH MANY MOLECULES" in line:
            return

        if "COMPUTATION TIME" in line:
            yield molecule_lines
            molecule_lines = []


def calculate(binary, filename, cwd=None):
    """
    Collect sets of lines for each molecule as they become availiable
    and then call a parser to extract the dictionary of properties.
    """
    calculations = run_mndo_file(binary, filename, cwd)

    props_list = []

    for mol_lines in calculations:
        props = get_properties(mol_lines)
        props_list.append(props)

    return props_list


def get_properties(lines):
    """
    Get properties of a single calculation.

    arguments:
        lines: list of MNDO output lines

    return:
        dict of properties

    NOTE some commented out sections from ppqm for other properties were
    removed for clarity, they can be restored from original code if we
    start doing multi-objective optimisation.
    """

    props = {}

    keywords = [
        # "CORE HAMILTONIAN MATR",  # high precision
        "ELECTRONIC ENERGY",  # low precision
        "NUCLEAR ENERGY",
        "IONIZATION ENERGY",
        "INPUT GEOMETRY",
    ]

    idx_keywords = get_rev_indices(lines, keywords)

    # SCF energy
    # NOTE it will always find "CORE HAMILTONIAN MATR" therefore a better
    # strategy might be to look for the second instance of "CORE HAMILTONIAN MATR"
    # rather than the last.
    # NOTE given we only get "NUCLEAR ENERGY" to 5 dp why do we not do the same
    # here for the "ELECTRONIC ENERGY"? we don't gain anything from using different
    # precisions given that we add them to get the net energy?

    # SCF energy - high precision
    # idx = idx_keywords[0]
    # if idx is None:
    #     print("\n".join(line))
    #     print("did not converge")
    #     props["e_scf"] = np.nan
    #     # exit()
    # else:
    #     idx -= 9
    #     line = lines[idx]
    #     # NOTE this an edge case correction
    #     if "SCF CONVERGENCE HAS BEE" in line:
    #         idx -= 2
    #         line = lines[idx]

    #     line = line.split()
    #     try:
    #         value = line[1]
    #         e_scf = float(value)
    #         props["e_scf"] = e_scf
    #     except IndexError:
    #         with open("failed", "w") as f:
    #             f.write("\n".join(lines))
    #         print("did not converge check failed for output")
    #         exit()

    # SCF energy - low precision
    idx = idx_keywords[0]
    if idx is None:
        # with open("failed-scf", "w") as f:
        #     f.write("\n".join(lines))
        # print("check failed-scf")
        # exit()
        e_scf = np.nan
    else:
        line = lines[idx]
        line = line.split()
        value = line[2]
        e_scf = float(value)
    props["e_scf"] = e_scf  # ev

    # Nuclear energy
    idx = idx_keywords[1]
    if idx is None:
        # with open("failed-nuc", "w") as f:
        #     f.write("\n".join(lines))
        # print("check failed-nuc")
        # exit()
        e_nuc = np.nan
    else:
        line = lines[idx]
        line = line.split()
        value = line[2]
        e_nuc = float(value)
    props["e_nuc"] = e_nuc  # ev

    # ionization
    # idx = get_rev_index(lines, "IONIZATION ENERGY")
    idx = idx_keywords[2]
    if idx is None:
        # with open("failed-ion", "w") as f:
        #     f.write("\n".join(lines))
        # print("check failed-ion")
        # exit()
        e_ion = np.nan
    else:
        line = lines[idx]
        value = line.split()[-2]
        e_ion = float(value)  # ev
    props["e_ion"] = e_ion

    # eisol
    eisol = {}
    idxs = get_indices(lines, "EISOL", "IDENTIFICATION")
    for idx in idxs:
        line = lines[idx]
        line = line.split()
        atom = int(line[0])
        value = line[2]
        eisol[atom] = float(value)  # ev

    # # input coords
    # # idx = get_rev_index(lines, "INPUT GEOMETRY")
    idx = idx_keywords[3]
    idx += 6
    atoms = []
    coord = []
    j = idx
    idx_atm = 1
    idx_x = 2
    idx_y = 3
    idx_z = 4
    # continue until we hit a blank line
    while not lines[j].isspace() and lines[j].strip():
        line = lines[j].split()
        atoms.append(int(line[idx_atm]))
        x = line[idx_x]
        y = line[idx_y]
        z = line[idx_z]
        xyz = [x, y, z]
        xyz = [float(c) for c in xyz]
        coord.append(xyz)
        j += 1

    # calculate energy
    e_iso = [eisol[a] for a in atoms]
    e_iso = np.sum(e_iso)

    # total energy
    energy = e_nuc + e_scf - e_iso

    props["energy"] = energy

    return props


def set_params(param_list, param_keys, mean_params=None, scale_params=None, cwd=None):
    """
    Save the current model parameters to the mndo input file.
    """
    txt = ""

    params = {key[0]: {} for key in param_keys}
    for (atom_type, prop), param in zip(param_keys, param_list):
        params[atom_type][prop] = param

    # reparameterise according to prior
    if mean_params and scale_params:
        for atomtype in params:
            p, s, d = params[atomtype], scale_params[atomtype], mean_params[atomtype]
            for key in p:
                val = p[key] * s[key] + d[key]
                txt += f"{key:8s} {atomtype:2s} {val:15.11f}\n"
    else:
        for atomtype in params:
            p = params[atomtype]
            for key in p:
                val = p[key]
                txt += f"{key:8s} {atomtype:2s} {val:15.11f}\n"

    param_file = "fort.14"

    if cwd is not None:
        cwd = fix_dir_name(cwd)
        param_file = cwd + param_file

    with open(param_file, "w") as f:
        f.write(txt)


def write_tmp_optimizer(atoms, coords, filename, method):

    txt = get_inputs(atoms, coords, np.zeros_like(atoms), range(len(atoms)), method)

    with open(filename, "w") as f:
        f.write(txt)


def get_inputs(atoms_list, coords_list, charges, titles, method):
    inptxt = ""
    for atoms, coords, charge, title in zip(atoms_list, coords_list, charges, titles):
        inptxt += get_input(atoms, coords, charge, title, method=method)

    return inptxt


def get_input(atoms, coords, charge, title, method):
    header = (
        f"{method} 1SCF MULLIK PRECISE charge={charge} iparok=1 jprint=5\n"
        "nextmol=-1\n"
        f"TITLE {title}\n"
    )
    txt = header

    for atom, coord in zip(atoms, coords):
        line = "{:2s} {:} 0 {:} 0 {:} 0\n".format(atom, *coord)
        txt += line

    return txt + "\n"


# Parallel code additions


def fix_dir_name(name):

    if not name.endswith("/"):
        name += "/"

    return name


def worker(*args, **kwargs):
    """
    """
    scr = kwargs["scr"]
    filename = kwargs["filename"]
    param_keys = kwargs["param_keys"]
    mean_params = kwargs["mean_params"]
    scale_params = kwargs["scale_params"]
    binary = kwargs["binary"]

    # Ensure unique directory
    scr = fix_dir_name(scr)
    pid = os.getpid()
    cwd = f"{scr}{pid}/"

    if not os.path.exists(cwd):
        os.mkdir(cwd)

    if not os.path.exists(cwd + filename):
        shutil.copy2(filename, cwd + filename)

    # Set params in worker dir
    param_list = args[0]
    set_params(param_list, param_keys, mean_params, scale_params, cwd=cwd)

    # Calculate properties
    properties_list = calculate(binary, filename, cwd=cwd)

    shutil.rmtree(cwd)

    return properties_list


def calculate_parallel(
    params_joblist,
    param_keys,
    mean_params,
    scale_params,
    filename,
    binary,
    n_procs=2,
    mndo_input=None,
    **kwargs,
):
    scr = "_tmp_mndo_/"
    if not os.path.exists(scr):
        os.mkdir(scr)

    worker_kwargs = {
        "scr": scr,
        "filename": filename,
        "param_keys": param_keys,
        "mean_params": mean_params,
        "scale_params": scale_params,
        "binary": binary,
    }

    mapfunc = partial(worker, **worker_kwargs)

    p = mp.Pool(n_procs)
    # results = p.map(mapfunc, params_joblist)
    results = list(tqdm(p.imap(mapfunc, params_joblist), total=len(params_joblist)))

    return results


# Utilities for extracting default parameters

# fmt: off
ATOMS = [
    "h", "he",
    "li", "be", "b", "c", "n", "o", "f", "ne",
    "na", "mg", "al", "si", "p", "s", "cl", "ar",
    "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu",
    "zn", "ga", "ge", "as", "se", "br", "kr",
    "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag",
    "cd", "in", "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy",
    "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt",
    "au", "hg", "tl", "pb", "bi", "po", "at", "rn",
    "fr", "ra", "ac", "th", "pa", "u", "np", "pu"
]
# fmt: on


def convert_atom(atom, t=None):
    """
    convert atom from str2int or int2str
    """

    if t is None:
        t = type(atom)
        t = str(t)

    if "str" in t:
        atom = atom.lower()
        idx = ATOMS.index(atom) + 1
        return idx

    else:
        atom = ATOMS[atom - 1].capitalize()
        return atom


def dump_default_params():
    """
    Function takes the default parameters from different methods in mndo
    and saves them output files.
    """
    # dump parameters
    methods = ["MNDO", "AM1", "PM3", "OM2"]

    for method in methods:
        params = get_default_params(method)
        filename = "parameters-{:}.json".format(method.lower())
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)


def get_default_params(binary, method):
    """
    Get the default parameters of a method
    """
    atoms = ["H", "C", "N", "O", "F"]
    # TODO: test if nextmol = -1 line is breaking
    # header = f"{method} 0SCF MULLIK PRECISE charge={{:}} jprint=5\n\nTITLE {{:}}"
    n_atoms = len(atoms)

    coords = np.arange(n_atoms * 3).reshape((n_atoms, 3))
    coords *= 5

    txt = get_input(atoms, coords, 0, "get params", method=method)
    filename = "_tmp_params.inp"

    with open(filename, "w") as f:
        f.write(txt)

    molecules = run_mndo_file(binary, filename)

    lines = next(molecules)

    idx = get_index(lines, "PARAMETER VALUES USED IN THE CALCULATION")
    idx += 4

    params = {}

    while True:

        line = lines[idx]
        line = line.strip().split()

        if len(line) == 0:

            line = lines[idx + 1]
            line = line.strip().split()

            if len(line) == 0:
                break
            else:
                idx += 1
                continue

        atom, param, value = line[0], line[1], float(line[2])
        # unit = line[3]
        # desc = " ".join(line[4:])

        atom = convert_atom(int(atom))

        if atom not in list(params.keys()):
            params[atom] = {}

        params[atom][param] = value

        idx += 1

    return params


if __name__ == "__main__":
    print("This is just a script of functions")
    pass
