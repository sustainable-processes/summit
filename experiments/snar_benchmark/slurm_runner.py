from summit import NeptuneRunner, get_summit_config_path
from paramiko import SSHClient
from scp import SCPClient
import uuid
import pathlib
import os

class SlurmRunner(NeptuneRunner):
    """  Run an experiment on a remote server (e.g., HPC) using SLURM.
    
    You need to set the environmental variables SSH_USER and SSH_PASSWORD
    with the information to log into the remote server. 

    This inherits NeptuneRunner so it will report up to Neptune.

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

        self.docker_container = kwargs.get('docker_container', 
                            "marcosfelt/summit:snar_benchmark")
        self.hostname = kwargs.get('hostname',"login-cpu.hpc.cam.ac.uk")

    def run(self, **kwargs):
        # Set up file structure
        base = pathlib.Path(".snar_benchmark")
        uuid_val = str(uuid.uuid4())
        save_file_dir = base /  uuid_val
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
        
        # SSH into remote server
        username = os.getenv('SSH_USER')
        password = os.getenv('SSH_PASSWORD')
        neptune_api_token = os.getenv('NEPTUNE_API_TOKEN')
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.hostname, username=username, password=password)

        # Make the .snar_benchmark folder on the remote server if it doesn't exist
        remote_path = f".snar_benchmark/{uuid_val}"
        ssh.exec_command(f"mkdir -p {remote_path}")

        # Copy files onto remote server
        scp = SCPClient(ssh.get_transport())
        scp.put([str(python_file_path), str(json_file_path), "slurm_summit_snar_experiment.sh"], 
                remote_path=remote_path)

        # Run the experiment
        ssh.exec_command(f"export NEPTUNE_API_TOKEN={neptune_api_token}")
        ssh.exec_command(f"cd {remote_path} && sbatch slurm_summit_snar_experiment.sh {self.docker_container} {python_file_path}")

        # Close the ssh connection
        scp.close()
