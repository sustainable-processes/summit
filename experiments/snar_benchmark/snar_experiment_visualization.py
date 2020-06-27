from summit import Runner

from neptune.sessions import Session, HostedNeptuneBackend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import  zipfile
import shutil
import warnings



class PlotExperiments:
    def __init__(self, project: str, experiment_ids: list):
        self.session = Session(backend=HostedNeptuneBackend())
        self.proj = self.session.get_project(project)
        self.runners = {}
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
            self.runners[experiment.id] = r

            # Remove file
            
            shutil.rmtree(f'data/{experiment.id}')

    def _create_param_df(self):
        """Create a parameters dictionary"""
        # Transform
        # Strategy
        # Batch Size
        records = []
        for experiment_id, r in self.runners.items():
            record = {}
            record['experiment_id'] = experiment_id

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
        self.df = pd.DataFrame.from_records(records)
        return self.df

    def iterations_to_threshold(self, sty_threshold = 1e4, 
        e_factor_threshold = 10.0):
        #Separate experiments
        df = self.df.copy()
        df = df.set_index('experiment_id')
        uniques = df.drop_duplicates(keep='last')
        df_new = self.df.copy()
        experiments = {}
        for i, unique in uniques.iterrows():
            temp_df = df_new.merge(unique.to_frame().transpose(), how='inner')
            name = f"{unique['strategy_name']}, {unique['transform_name']}, batch size={unique['batch_size']}"
            #Create dictionary with experiment names as keys and array of repeats of experiment_id as values
            experiments[name] = temp_df['experiment_id'].values

        # Find iterations to thressodl
        results = []
        for name, ids in experiments.items():
            num_iterations = []
            something_happens = False
            for i, experiment_id in enumerate(ids):
                data = self.runners[experiment_id].experiment.data[['sty','e_factor']]
                meets_threshold = data[(data['sty']>=sty_threshold) & (data['e_factor']<=e_factor_threshold)]
                if len(meets_threshold.index) > 0:
                    num_iterations.append(meets_threshold.index[0])
                    something_happens = True

            if something_happens:
                mean = np.mean(num_iterations)
                std = np.std(num_iterations)
                results.append(dict(name=name, mean_iterations=mean, std_iterations=std))
            else:
                warnings.warn("Cannot find any iterations that achieve threshold")
        
        return pd.DataFrame(results)



        # data_to_plot = []
        # j = 0
        # for name, ids in experiments.items(): 
        #     data = np.empty([len(ids), max_iterations])
        #     for i, expriment_id in enumerate(ids):
        #         datum = self.runners[expriment_id].experiment.data[data_column].values
        #         for d in datum:
        #             pass
        #         data[i, 0:len(datum)] = datum

    
            
        #     #Calculate means
        #     means = np.nanmean(data, axis=0)
        #     stds = np.nanstd(data, axis=0)
            
        #     #plots
        #     x = np.arange(1, data.shape[1]+1)
        #     ax.plot(x, means, label=name, color=colors[j])
        #     ax.fill_between(x,means-stds, means+stds, 
        #                     alpha=0.1, color=colors[j])
        #     j+= 1
        # ax.legend()
        # return ax



            


        
                
        