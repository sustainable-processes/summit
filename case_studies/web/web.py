from summit.data import DataSet
from summit.domain import ContinuousVariable, Constraint, Domain
from summit.strategies import TSEMO2
from summit.models import GPyModel

import pandas as pd

from anvil import media, server, tables
from anvil.tables import app_tables

import os
from dotenv import load_dotenv
load_dotenv()

@server.callable
def get_suggestions(project, experiments, num_experiments):
    print("New request received")
    variables = app_tables.variable.search(project=project)

    #Set up domain
    domain = Domain()
    for v in variables:
        domain += ContinuousVariable(name=v.get_id(),
                                     description=v['description'],
                                     bounds=(v['lower_bound'],v['upper_bound']),
                                     is_objective=v['is_objective'],
                                     maximize=v['maximize'])
    
    #Get experimental data into a dataset
    experiment_data = [experiment['data'] for experiment in experiments]
    df = pd.DataFrame(experiment_data)
    data = DataSet.from_df(df)
    
    #Set up models
    num_outputs = len(domain.output_variables)
    num_inputs = domain.num_variables()
    models = {}
    for v in domain.variables:
        if v.is_objective:
            models[str(v.name)] = GPyModel(input_dim=num_inputs)

    print("Running optimization")
    #Run the optimization
    tsemo = TSEMO2(domain=domain, models=models)
    results =  tsemo.generate_experiments(previous_results=data, num_experiments=3)
    
    print("Returning result")
    #Convert from dataset to json list
    results_list = results.to_dict(orient='records')
    new_results_list = []
    for result in results_list:
        new_results_dict = {}
        for key, value in result.items():
            new_results_dict[key[0]] =  value
        new_results_list.append(new_results_dict)
    new_results_list
    
    return new_results_list

if __name__ == '__main__':
    anvil_key = os.getenv('ANVIL_UPLINK_KEY')
    print("Connecting to Anvil")
    server.connect(anvil_key)
    print("Starting server")
    try:
        server.wait_forever()
    except KeyboardInterrupt:
        print("Stopping Server")