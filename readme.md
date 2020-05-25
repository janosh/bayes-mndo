# fitting

This repo contains code and datasets for reparametrizing SQM methods.

## License:

All datasets (including input files, log files, csv files, xyz structure files, etc) are licensed under the CC0 1.0 Universal license.

Scripts and other code is licensed under the MIT License.

## Environment

The environment file `env.yml` was generated with `conda env export --no-builds > env.yml`. To recreate the environment from this file run `conda env create -f env.yml`.

The environment `mndo` was originally created by running the command:

```sh
conda create -n mndo python=3.6 \
  && conda activate mndo \
  && pip install tensorflow tensorflow-probability rmsd \
    pandas scikit-learn tqdm matplotlib pre-commit flake8 black notebook plotly
```

To delete the environment run `conda env remove -n mndo`.

To update all packages and reflect changes in this file use

```sh
conda update --all \
  && pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U \
  && conda env export --no-builds > env.yml
```

## Code Health

To install `git` pre-commit hooks for `black`, `flake8` and `isort`, run `pre-commit install` from the terminal. Recommended prior to contributing.
