import copy
import functools
import json
import multiprocessing as mp
import os
import shutil
import subprocess

import numpy as np

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


def get_indices(lines, pattern, stop_pattern=None):

    idxs = []

    for i, line in enumerate(lines):
        if pattern in line:
            idxs.append(i)
            continue

        if stop_pattern and stop_pattern in line:
            break

    return idxs


def get_index(lines, pattern):
    for i, line in enumerate(lines):
        if pattern in line:
            return i
    return None


def reverse_enum(lst):
    for index in reversed(range(len(lst))):
        yield index, lst[index]


def get_rev_indices(lines, patterns):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None] * n_patterns

    for i, line in reverse_enum(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs


def execute(cmd, filename, cwd):
    """
    Call the MNDO fortran binary. For this function to work requires `mndo` to be in path.
    """
    
    if cwd is not None:
        job_cmd = f"cd {cwd}; {cmd} < {filename}"
    else:
        job_cmd = f"{cmd} < {filename}"

    popen = subprocess.Popen(
        job_cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_mndo_file(filename, cwd=None):
    """
    Runs mndo on the given input file and yields groups of lines for each
    molecule as the program completes.
    """
    cmd = "/home/reag2/PhD/second-year/fitting/mndo/mndo99_binary"
    cmd = os.path.expanduser(cmd)
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

    print("lines:", list(lines))


def calculate(filename, cwd=None):
    """
    Collect sets of lines for each molecule as they become availiable
    and then call a parser to extract the dictionary of properties.
    """
    calculations = run_mndo_file(filename, cwd)

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
    """

    props = {}

    keywords = [
        "CORE HAMILTONIAN MATR",
        "NUCLEAR ENERGY",
        "IONIZATION ENERGY",
        "INPUT GEOMETRY",
    ]

    idx_keywords = get_rev_indices(lines, keywords)

    # SCF energy
    idx = idx_keywords[0]
    idx -= 9
    line = lines[idx]
    if "SCF CONVERGENCE HAS BEE" in line:
        idx -= 2
        line = lines[idx]

    line = line.split()
    value = line[1]
    e_scf = float(value)
    props["e_scf"] = e_scf

    # Nuclear energy
    idx = idx_keywords[1]
    line = lines[idx]
    line = line.split()
    value = line[2]
    e_nuc = float(value)
    props["e_nuc"] = e_nuc  # ev

    # eisol
    eisol = {}
    idxs = get_indices(lines, "EISOL", "IDENTIFICATION")
    for idx in idxs:
        line = lines[idx]
        line = line.split()
        atom = int(line[0])
        value = line[2]
        eisol[atom] = float(value)  # ev

    # ionization
    # idx = get_rev_index(lines, "IONIZATION ENERGY")
    idx = idx_keywords[2]
    line = lines[idx]
    value = line.split()[-2]
    e_ion = float(value)  # ev
    props["e_ion"] = e_ion

    # input coords
    # idx = get_rev_index(lines, "INPUT GEOMETRY")
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
    energy = e_nuc + e_scf - e_iso

    props["energy"] = energy

    return props


@functools.lru_cache()
def load_prior_dicts(
    scale_path="../parameters/scale-pm3.json",
    default_path="../parameters/parameters-pm3.json",
):

    with open(default_path, "r") as file:
        raw_json = file.read()
        default_dict = json.loads(raw_json)

    with open(scale_path, "r") as file:
        raw_json = file.read()
        scale_dict = json.loads(raw_json)
    #     default_opt_dict = json.loads(raw_json)

    # scale_dict = {}
    # for atomtype in default_dict:
    #     scale_dict[atomtype] = {}
    #     atom, opt_atom = default_dict[atomtype], default_opt_dict[atomtype]
    #     for key in default_dict[atomtype]:
    #         scale_dict[atomtype][key] = atom[key] - opt_atom[key]

    return default_dict, scale_dict


def set_params(params, cwd=None):
    """
    Save the current model parameters to the mndo input file.
    """
    txt = ""
    defaults, scales = load_prior_dicts()

    for atomtype in params:
        p, s, d = params[atomtype], scales[atomtype], defaults[atomtype]
        for key in p:
            val = p[key] * s[key] + d[key]
            txt += f"{key:8s} {atomtype:2s} {val:15.11f}\n"

    filename = "fort.14"

    if cwd is not None:
        cwd = fix_dir_name(cwd)
        filename = cwd + filename

    with open(filename, "w") as file:
        file.write(txt)


def write_tmp_optimizer(atoms, coords, method, filename="_tmp_optimizer"):

    txt = get_inputs(atoms, coords, np.zeros_like(atoms), range(len(atoms)), method)

    with open(filename, "w") as f:
        f.write(txt)


def get_inputs(atoms_list, coords_list, charges, titles, method=None):
    """
    """
    inptxt = ""
    for atoms, coords, charge, title in zip(atoms_list, coords_list, charges, titles):
        inptxt += get_input(atoms, coords, charge, title, method=method)

    return inptxt


def get_input(atoms, coords, charge, title, method=None):
    """
    """
    header = (
        f"{method or 'OM2'} 1SCF MULLIK PRECISE charge={charge} iparok=1 jprint=5\n"
        "nextmol=-1\n"
        f"TITLE {title}\n"
    )
    txt = header

    for atom, coord in zip(atoms, coords):
        line = "{:2s} {:} 0 {:} 0 {:} 0\n".format(atom, *coord)
        txt += line

    return txt + "\n"

## Parallel code additions

def get_pinfo():
    """
    get process id of parent and current process
    """
    ppid = os.getppid()
    pid = os.getpid()
    return ppid, ppid


def fix_dir_name(name):

    if not name.endswith("/"):
        name += "/"

    return name


def get_indexes_patterns(lines, patterns):
    
    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None]*n_patterns

    for i, line in enumerate(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs


def worker(*args, **kwargs):
    
    scr = kwargs["scr"]
    filename = kwargs["filename"]

    # Ensure unique directory
    scr = fix_dir_name(scr)
    pid = os.getpid()
    cwd = f"{scr}{pid}/"

    if not os.path.exists(cwd):
        os.mkdir(cwd)

    if not os.path.exists(cwd + filename):
        shutil.copy2(scr + filename, cwd + filename)

    # Set params in worker dir
    params = args[0]
    set_params(params, cwd=cwd)

    # Calculate properties
    properties_list = calculate(filename, cwd=cwd)


def calculate_multi_params(
    inputstr,
    params_list,
    scr=None,
    n_procs=1):
    """
    """

    scr = "_tmp_mndo_/"
    if not os.path.exists(scr):
        os.mkdir(scr)

    filename = "_tmp_inputstr_"
    with open(scr + filename, 'w') as f:
        f.write(inputstr)

    kwargs = {"scr": scr, "filename": filename,}

    mapfunc = functools.partial(worker, **kwargs)

    p = mp.Pool(n_procs)
    results = p.map(mapfunc, params_list)

    return results


def get_tmp_optimizer(atoms, coords, method, filename="_tmp_optimizer"):
    
    txt = get_inputs(atoms, coords, np.zeros_like(atoms), range(len(atoms)), method)

    return txt


def numerical_jacobian(inputstr, param_vals, param_keys, dh=10**-5, n_procs=2):
    """
    get properties for
    """

    params_joblist = []

    params = {key[0]: {} for key in param_keys}
    param_grad = {key[0]: {} for key in param_keys}
    for (atom_type, prop), param in zip(param_keys, param_vals):
        params[atom_type][prop] = param
        param_grad[atom_type][prop] = []

    for (atom_type, prop) in param_keys:
        dparams = copy.deepcopy(params)

        # forward
        dparams[atom_type][prop] += dh
        params_joblist.append(copy.deepcopy(dparams))

        # backward
        dparams[atom_type][prop] -= 2*dh
        params_joblist.append(copy.deepcopy(dparams))


    # Calculate all results
    results = calculate_multi_params(inputstr, params_joblist, n_procs=n_procs)
    n_results = len(results)

    i = 0
    for atom in params.keys():
        for key in params[atom].keys():
            param_grad[atom][key].append(results[i])
            param_grad[atom][key].append(results[i+1])
            i += 2

    print(param_grad)

    return param_grad

## Utilities for extracting default parameters


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


def get_default_params(method):
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

    molecules = run_mndo_file(filename)

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
