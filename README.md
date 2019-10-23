# Summit

Summit is a set of tools for optimizing chemical processes. 

## Installation

pip install summit --upgrade --extra-index-url pypi.rxns.io

## Case Studies

The best way to understand how to use summit is to take a look at the case studies.  These contain jupyter notebook with detailed examples to follow. 

* [Formulation](case_studies/formulation)
* [Nanosilica](case_studies/nanosilica)
* [Photo Amination](case_studies/photoamination/)
* [Borrowing Hydrogen](case_studies/borrowing_hydrogen)

## Develpment

### Build a release

1. Install [s3pypi](https://github.com/novemberfiveco/s3pypi) and [dephell](https://dephell.org/docs/installation.html)
2. Install AWS credentials to upload pypi.rxns.io (Kobi is the one who controls this).
3. Bump the version in pyproject.toml and then run:
    ```dephell deps convert --from=pyproject.toml --to=setup.py```
4. Go into setup.py and delete the lines for extras_install_requires
4. Upload the package to the private pypi repository:
    ```s3pypi --bucket pypi.rxns.io```


### Building Containers

This is useful when you want to run summit on a remote server. 

1. Download [habitus](https://www.habitus.io/). Habitus is used to insert ssh keys into containers safely, so you can pull from for private Github repositories. 

2. If you are on mac, you need to install gnu-tar:

    ```
    brew install gnu-tar
    echo "export PATH=/usr/local/opt/gnu-tar/libexec/gnubin:$PATH" >> ~./bash_profile
    echo "aliast tar='gtar'" >> ~/.bash_profile
    source ~/.bash_profile
    ```

3. Change line 10 of the build.yml to be the name of your [ssh key for github](https://help.github.com/en/articles/connecting-to-github-with-ssh) 

4. Inside the main folder for summit (i.e., the same folder as this README), build the container:

    ```
    #On Mac
    sudo habitus --build host=docker.for.mac.localhost --binding=127.0.0.1 --secrets=true

    #This is what I think it will be on linux
    sudo habitus --build host=gateway.docker.local --binding=127.0.0.1 --secrets=true
    ```

