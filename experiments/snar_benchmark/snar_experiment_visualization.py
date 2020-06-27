import neptune
from neptune.sessions import Session, HostedNeptuneBackend
from summit import Runner
import os
import  zipfile
import shutil


class PlotExperiments:
    def __init__(self, project: str):
        self.session = Session(backend=HostedNeptuneBackend())
        self.proj = self.session.get_project(project)
        self.runners = []

    def restore_runners(self, experiment_ids: list):
        # Download artifacts    
        experiments = self.proj.get_experiments(id=experiment_ids)
        for experiment in experiments:
            path = f'data/{experiment.id}'
            try:
                os.mkdir(path,)
            except FileExistsError:
                pass
            experiment.download_artifacts(destination_dir=path)

            # Unzip somehow 
            files = os.listdir(path)
            with zipfile.ZipFile(path+'/'+files[0], 'r') as zip_ref:
                zip_ref.extractall(path)

            # Get filename
            path += '/output'
            files = os.listdir(path)
            files_json = [f for f in files if '.json' in f]

            # Restore runner
            self.runners.append(Runner.load(path+'/'+files_json[0]))

            # Remove file
            
            shutil.rmtree(f'data/{experiment.id}')

    def plot_repeats(self, variable_name):
        """Plot repeats of the same settings"""
        pass

    