import neptune
from neptune.sessions import Session, HostedNeptuneBackend
from summit import Runner
import os
import  zipfile
import shutil
import pandas as pd


class PlotExperiments:
    def __init__(self, project: str, experiment_ids: list):
        self.session = Session(backend=HostedNeptuneBackend())
        self.proj = self.session.get_project(project)
        self.runners = []
        self.experiment_ids = experiment_ids
        self._restore_runners()
        self._create_param_df()

    def _restore_runners(self):
        """Restore runners from Neptune Artifacts"""
        # Download artifacts    
        experiments = self.proj.get_experiments(id=self.experiment_ids)
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
            r = Runner.load(path+'/'+files_json[0])
            r.experiment_id = experiment.id
            self.runners.append(r)

            # Remove file
            
            shutil.rmtree(f'data/{experiment.id}')

    def _create_param_df(self):
        """Create a parameters dictionary"""
        # Transform
        # Strategy
        # Batch Size
        records = []
        for r in self.runners:
            record = {}
            record['experiment_id'] = r.experiment_id

            # Transform
            transform_name = r.strategy.transform.__class__.__name__
            transform_params = r.strategy.transform.to_dict()['transform_params']
            record['transform_name'] = transform_name
            record.update(transform_params)

            # Strategy
            record['strategy_name'] = r.strategy.__class__.__name__

            # Batch size
            record['batch_size'] = r.batch_size

            records.append(record)

        #Make pandas dataframe
        df = pd.DataFrame.from_records(records)
        self.df = df.set_index('experiment_id')
        return df