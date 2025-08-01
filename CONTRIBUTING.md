# Contributing Guidelines

:tada: **First off, thank you for considering contributing to TICOI! ** :tada:

The project can still be improved, and all contribitors are welcome!
These are some of the many ways to contribute:

* :bug: Submitting bug reports and feature requests
* :memo: Writing tutorials or examples
* :mag: Fixing typos and improving the documentation
* :bulb: Writing code for everyone to use

*
Below is a guide to contributing to TICOI step by step, ensuring tests are passing.

## Overview: making a contribution

The technical steps to contributing to xDEM are:

1. Fork `ticoi/ticoi` and clone your fork repository locally.
2. Set up the development environment **(see section "Setup" below)**,
3. Create a branch for the new feature or bug fix,
4. Make your changes,
5. Add or modify related tests in `tests/` **(see section "Tests" below)**,
6. Commit your changes,
7. Run `pre-commit` separately if not installed as git hook **(see section "Linting" below)**,
8. Push to your fork,
9. Open a pull request from GitHub to discuss and eventually merge.

## Development environment

TICOI currently supports Python versions of 3.10 to 3.11, which are
tested in a continuous integration (CI) workflow running on GitHub Actions.

When you open a PR on TICOI, a single linting action and 1 test actions will automatically start, corresponding to all
supported Python versions (3.11).

### Setup

#### With `mamba`
Clone the git repo and create a `mamba` environment (see how to install `mamba` in the [mamba documentation](https://mamba.readthedocs.io/en/latest/)):

```bash
git clone git@github.com:ticoi/ticoi.git
cd ticoi
mamba env create -f environment.yml -n ticoi_env  # change the name if you want
mamba activate ticoi_env  # Or any other name specified above
```
#### With `pip`
```bash
git clone git@github.com:ticoi/ticoi.git
cd ticoi
make install
```

### Tests

At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, using `pytest` (see existing tests for examples).
The structure of test modules and functions in `tests/` largely mirrors that of the package modules and functions in `ticoi/`.

To run the entire test suite, run `pytest` from the root of the repository:
```bash
pytest
```
### Formatting and linting

Install and run `pre-commit` from the root of the repository (such as with `mamba install pre-commit`, see [pre-commit documentation](https://pre-commit.com/) for details),
which will use `.pre-commit-config.yaml` to verify spelling errors, import sorting, type checking, formatting and linting:

```bash
pre-commit install #optional: to install pre-commit as a git-hook, to ensure checks have to pass before committing.
pre-commit run --all
```

You can then commit and push those changes.

### Final steps

That's it! If the tests are passing, or if you need help to make those work, you can open a PR.

We'll receive word of your PR as soon as it is opened, and should follow up shortly to discuss the changes, and eventually give approval to merge. Thank you so much for contributing!

### Rights

The license (see LICENSE) applies to all contributions.


### Understanding the structure of the code

#### Main code

* **core.py**: Main functions to process the temporal inversion of glacier's surface velocity using
  the TICOI method. The inversion is solved using an Iterative Reweighted Least Square, and a robust downweighted
  function (Tukey's biweight).
* **cube_data_classxr.py**: Class object to store and manipulate velocity observation data in a cube (netcdf or zarr)
* **pixel_class.py**: Class object to manipulate and visualize velocity observations and inverted results on a pixel (from a pandas dataframe, or inside a cube)
* **inversion_functions.py**: Functions to process the temporal inversion.
* **interpolation_functions.py**: Functions to process the temporal interpolation.
* **filtering_functions.py**: Functions to process some filtering.
* **utils.py**: Two other functions for accessing ITS_LIVE data.
* **mjd2date.py**: Functions to convert the dates from Modified Julian Date to Gregorian Date



