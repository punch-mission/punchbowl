# punchbowl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14029123.svg)](https://doi.org/10.5281/zenodo.14029123)

`punchbowl` is the science calibration code for [the PUNCH mission](https://punch.space.swri.edu/).

### [Start by checking the documentation.](https://punchbowl.readthedocs.io/en/latest/)

> [!CAUTION]
> This package is still being heavily edited as calibration algorithms are improved.
> Stability is not promised until v1.

## Accessing the data

Data are available via the Solar Data Analysis Center.
See [the PUNCH website](https://punch.space.swri.edu/punch_science_getdata.php) for details.

## Installing `punchbowl`

Install with `pip install punchbowl` to get the released version.

To get the latest unreleased version: clone the repo and install it locally.

## Setting up the PUNCH environment

### Using uv
1. Clone this repository on your local machine
2. Navigate to the repository and create a virtual environment in Python using `uv` with `uv sync --no-sources --no-dev`
   (super users who want to install optional-dependencies, use `uv sync --no-sources --no-dev --all-extras`)
3. Activate the virtual environment by running `source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows
4. Explore some data!

### Using venv
1. Clone this repository on your local machine
2. Navigate to the repository and create a virtual environment in Python using `python -m venv my_venv_name`
3. Activate the virtual environment by running `source my_venv_name/bin/activate` on Mac/Linux or `my_venv_name\Scripts\activate` on Windows
4. Install the project environment dependencies by running `pip install .`
   (super users who want to install optional-dependencies, use `pip install -e ".[super-user]"`)
5. Explore some data!


## Running `punchbowl`

[The documentation](https://punchbowl.readthedocs.io/en/latest/index.html) provides details on how to run the various components.
It also provides a short explanation of each underlying algorithm.
Please reach out with a discussion for more help.

## Testing

You need Docker or Podman Desktop.

1. Install Podman Desktop using your preferred method
2. Pull the mariadb image with podman pull docker.io/library/mariadb
3. Run tests with pytest


## Getting help

Please open an issue or discussion on this repo.

## Contributing
gi
We appreciate all contributions.
If you have a problem with the code or would like to see a new feature, please open an issue.
Or you can submit a pull request.

If you would like to contribute by submitting a pull request, ensure that you have installed the environment as a super-user to activate necessary dependencies.
You will also need to set up pre-commit so that it will analyze your staged files.
To do this, run `pre-commit install` in the punchbowl, solpolpy, and/or regularizepsf repositories after setting up your environment.
This step should only need to be done once after installing the environment.

Thanks to all the contributors to punchbowl!

<a href="https://github.com/punch-mission/punchbowl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=punch-mission/punchbowl" />
</a>
