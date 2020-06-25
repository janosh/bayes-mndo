import multiprocessing as mp
import os
import shutil
from functools import partial

from tqdm import tqdm

import data
from chemhelp import mndo

# def calculate(binary, filename, scr=None):
#     """
#     Collect sets of lines for each molecule as they become available
#     and then call a parser to extract the dictionary of properties.

#     DEPRECIATED

#     """
#     props_list = mndo.calculate_file(filename, scr=scr, mndo_cmd=binary)
#     props_list = list(props_list)  # NOTE that calculate_file returns an iterator

#     return props_list


def calculate_parallel(
    params_joblist,
    param_keys,
    mean_params,
    scale_params,
    filename,
    binary,
    n_procs=2,
    mndo_input=None,
    scr="_tmp_optim",
    **kwargs,
):

    worker_kwargs = {
        "scr": scr,
        "filename": filename,
        "param_keys": param_keys,
        "mean_params": mean_params,
        "scale_params": scale_params,
        "binary": binary,
    }

    mapfunc = partial(worker, **worker_kwargs)

    # NOTE generating multiple pools each iteration was leading to a memory leak
    # NOTE using imap may be slower but done for development purposes to check
    # it's working
    with mp.Pool(n_procs) as p:
        # results = p.map(mapfunc, params_joblist)
        results = list(tqdm(p.imap(mapfunc, params_joblist), total=len(params_joblist)))

    return results


def worker(*args, **kwargs):
    """
    """
    scr = kwargs["scr"]
    filename = kwargs["filename"]
    param_keys = kwargs["param_keys"]
    mean_params = kwargs["mean_params"]
    scale_params = kwargs["scale_params"]
    binary = kwargs["binary"]

    # Ensure unique directory for this worker in scratch directory
    pid = os.getpid()
    cwd = os.path.join(scr, str(pid))

    if not os.path.exists(cwd):
        os.mkdir(cwd)

    if not os.path.exists(os.path.join(cwd, filename)):
        shutil.copy2(os.path.join(scr, filename), os.path.join(cwd, filename))

    # Set params in worker dir
    param_list = args[0]
    data.set_params(
        param_list, param_keys, mean_params, scale_params, scr=cwd,
    )

    # Calculate properties
    properties_list = mndo.calculate_file(filename, scr=cwd, mndo_cmd=binary)

    # NOTE JCK properties_list is a generator, so complete parsing on worker
    properties_list = list(properties_list)

    shutil.rmtree(cwd)

    return properties_list
