# Experiments

This is the code used to run and analyze the experiments in the paper.  Each folder has code to run experiments, some code for visualization and a Jupyter notebook that contains all the plots used in the paper. 

## Installation

To run the experiments code, you will need to install the extra experiments dependencies:

```bash
pip install summit[experiments]
```

## Steps to run on the HPC

1. Commit changes and push to Github
2. Build the Docker container and push to docker hub. I do this on our private server since it requires quite a bit of space.

    ```
    docker build . -t marcosfelt/summit:tag
    docker push marcosfelt/summit:tag
    ```
    Replace `tag` with the name of the branch.

3. Log into the HPC and pull the container using singularity. It's important to do this, so each experiment doesn't have to pull the container.

    ```
    singularity run docker://marcosfelt/summit:tag
    ```
    Replace `tag` with the tag you used in step 2.

4. Run the test script. For the C-N benchmark for example:
    ```
    export SSH_USER= # put your HPC login username here
    export SSH_PASSWORD= # put your HPC login password here
    export NEPTUNE_API_TOKEN =  # put your Neptune API Token here
    poetry run pytest test_cn_experiment_MO.py
    ```
    The scripts automatically login to the HPC, submit the jobs to slurm and make sure the Neptune is setup for experiment tracking.