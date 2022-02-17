# Contributing

Some instructions for people contributing back.

### Downloading the code

1. Clone the repository:
```git clone https://github.com/sustainable-processes/summit_private.git```
2. Intall poetry by following the instructions [here](https://python-poetry.org/docs/#installation). We use poetry for dependency management.
3. Install all dependencies:
```poetry install```
3. To run tests:
```poetry run pytest --doctest-modules --ignore=case_studies```

### Commit Worfklow

- Use the [project board](https://github.com/orgs/sustainable-processes/projects/1) to keep track of issues. Issues will automatically be moved along in the board when they are closed in Github.
- Write tests in the tests/ folder
- Documentation follows the [numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-class-instances)
    - Please include examples when possible that can be tested using [doctest](https://docs.python.org/3/library/doctest.html)
    - All publicly available classes and methods should have a docstring
- Commit to a branch off master and submit pull requests to merge. 
    - To create a branch locally and push it:
    ```bash
    $ git checkout -b BRANCH_NAME
    # Once you've made some changes
    $ git commit -am "commit message"
    $ git push -u origin BRANCH_NAME
    #Now if you come back to Github, your branch should exist
    ```
    - All pull requests need one review.
    - Tests will be run automatically when a pull request is created, and all tests need to pass before the pull request is merged. 

### Docker
Sometimes, it is easier to run tests using a Docker container (e.g., on compute clusters). Here are the commands to build and run the docker containers using the included Dockferfile. The container entrypoint is python, so you just need to specify the file name.

To build the container and upload the container to Docker Hub.:
```
docker build . -t marcosfelt/summit:latest
docker push marcosfelt/summit:latest
```
You can change the tag from `latest` to whatever is most appropriate (e.g., the branch name). I have found that this takes up a lot of space on disk, so I have been running the commands on our private servers.

Then, to run a container, here is an example with the SnAr experiment code. The home directory of the container is called `summit_user`, hence we mount the current working directory into that folder.  We remove the container upon finishing using `--rm` and make it interactive using `--it` (remove this if you just want the container to run in the background). [Neptune.ai](https://neptune.ai/) is used for the experiments so the API token is passed in. Finally, I specify the image name and the tag and before referencing the python file I want to run. 

```
export NEPTUNE_API_TOKEN= #place your neptune token here
sudo docker run -v `pwd`/:/summit_user --rm -it --env NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN summit:snar_benchmark snar_experiment_2.py
```

Singularity (for running Docker containers on the HPC):
```
export NEPTUNE_API_TOKEN=
singularity exec -B `pwd`/:/summit_user docker://marcosfelt/summit:snar_benchmark snar_experiment.py
```