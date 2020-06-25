# Bayes MNDO

This repo contains code, datasets and binaries for reparametrizing MNDO. MNDO is a semi-empirical method for quantum calculations and stands for Modified Neglect of Diatomic Overlap.

## Environment

The environment file `env.yml` was generated with `conda env export --no-builds > env.yml`. To recreate the environment from this file run `conda env create -f env.yml`.

The environment `mndo` was originally created by running the command:

```sh
conda create -n mndo python=3.6 \
  && conda activate mndo \
  && pip install tensorflow tensorflow-probability rmsd \
    pandas scikit-learn tqdm matplotlib pre-commit flake8 black notebook plotly
```

To run `turbo_optim.py` you will also need the modified `TuRBO` package which can be installed with:

```sh
conda activate mndo
git clone https://github.com/CompRhys/TuRBO
cd TuRBO
python setup.py sdist
pip install -e .
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

## Troubleshooting

If you're trying to run one of the optimization scripts (`scipy`, `turbo`, `hmc`) and getting non-zero exit codes like

```sh
subprocess.CalledProcessError: Command '/path/to/your/mndo99_binary < _tmp_molecules' returned non-zero exit status 127.
```

that means the binary file could be found. In that case, check your path and try an absolute rather than relative path.

If you're getting the code 126, it's because the `mdno` command can be found but is not executable. In that case try `chmod +x /path/to/your/mndo99_binary`.
