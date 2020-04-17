# Summit

Summit is a set of tools for optimizing chemical processes. 

## Installation

If you want to use summit immediately without installing python on your computer, go to our [Jupyterhub](hub.rxns.io), which already has it installed. You can find a description of it [here](https://github.com/sustainable-processes/server/blob/master/notes/session_1.md).

To install locally:

```pip install git+https://github.com/sustainable-processes/summit.git@0.2.2#egg=summit```

You might need to enter your username and password for Github. 

## Documentation

The documentation for summit can be found on the [wiki](https://github.com/sustainable-processes/summit/wiki).

## Case Studies

In addition to the documentation, we are prepareing several case studies.  These contain jupyter notebooks with practical examples to follow. 

* [Formulation](case_studies/formulation)
* [Nanosilica](case_studies/nanosilica)
* [Photo Amination](case_studies/photoamination/)
* [Borrowing Hydrogen](case_studies/borrowing_hydrogen)

## Develpment

### Build a release

Below is the old process.

1. Install [s3pypi](https://github.com/novemberfiveco/s3pypi) and [dephell](https://dephell.org/docs/installation.html)
2. Install AWS credentials to upload pypi.rxns.io (Kobi is the one who controls this).
3. Bump the version in pyproject.toml and then run:
    ```dephell deps convert --from=pyproject.toml --to=setup.py```
4. Go into setup.py and delete the lines for extras_install_requires
4. Upload the package to the private pypi repository:
    ```s3pypi --bucket pypi.rxns.io```


