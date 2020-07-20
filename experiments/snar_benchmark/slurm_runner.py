from summit import NeptuneRunner, get_summit_config_path
import uuid
import pathlib
import os

class SlurmRunner(NeptuneRunner):
    """  Run a closed-loop strategy and experiment cycle
    
    Parameters
    ---------- 
    strategy : `summit.strategies.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : `summit.experiment.Experiment`
        The experiment class to use for running experiments. If None,
        the ExternalExperiment class will be used, which assumes that
        data from each experimental run will be added as a keyword
        argument to the `run` method.
    docker_container : str, optional
        The name of the docker container used by singularity.
        Defaults to marcosfelt/summit:snar_benchmark
    neptune_project : str
        The name of the Neptune project to log data to
    neptune_experiment_name : str
        A name for the neptune experiment
    netpune_description : str, optional
        A description of the neptune experiment
    files : list, optional
        A list of filenames to save to Neptune
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is 100.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments. Default is 1.
    f_tol : float, optional
        How much difference between successive best objective values will be tolerated before stopping.
        This is generally useful for nonglobal algorithms like Nelder-Mead. Default is None.
    hypervolume_ref : array-like, optional
        The reference for the hypervolume calculation if it is a multiobjective problem.
        Should be an array of length the number of objectives. Default is at the origin.
    Examples
    --------    
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.docker_container = kwargs.get('docker_container', "marcosfelt/summit:snar_benchmark")

    def run(self, **kwargs):
        # Set up file structure
        base = pathlib.Path(".snar_benchmark")
        save_file_dir = base /  str(uuid.uuid4())
        os.makedirs(save_file_dir, exist_ok=True)

        # Save json
        json_file_path = save_file_dir / "slurm_runner.json"
        self.save(json_file_path)

        # Create python file        
        python_file_path = save_file_dir / "run.py"
        with open(python_file_path, 'w') as f:
            f.write("from summit import NeptuneRunner\n")
            f.write(f"""r = NeptuneRunner.load("{json_file_path}")\n""")
            f.write("r.run(save_at_end=True)")
        
        # Run slurm job
        os.system(f"sbatch slurm_summit_snar_experiment.sh {self.docker_container} {python_file_path}")
