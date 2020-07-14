from summit.initial_design import LatinDesigner
from summit.data import DataSet
from summit.domain import ContinuousVariable, Constraint, Domain
from summit.strategies import TSEMO
from summit.models import GPyModel

import pandas as pd

from anvil import media, server, tables
from anvil.tables import app_tables

import os
import logging

# The ANVIL_UPLINK_KEY must be an environmental variable in .ennv

ENV = os.getenv("ENVIRONMENT", "DEV")

if ENV == "DEV":
    from dotenv import load_dotenv

    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    filename="myapp.log",
    format="%(asctime)s %(levelname)s:%(message)s",
)


@server.callable
def get_suggestions(project, experiments, num_experiments):
    try:
        project_id = project.get_id()
    except AttributeError:
        logging.error("No project found")
        return

    logging.debug(f"New request received from project {project_id}")
    variables = app_tables.variable.search(project=project)

    # Set up domain
    domain = Domain()
    maximize = None
    for v in variables:
        if v["is_objective"]:
            if v["objective_type"] in ["target", "minimize"]:
                maximize = False
            elif v["objective_type"] == "maximize":
                maximize = True
            else:
                raise ValueError(
                    f"Objective type is {v['objective_type']}. It must be minimize, maximize or target."
                )
        domain += ContinuousVariable(
            name=v.get_id(),
            description=v["description"],
            bounds=(v["lower_bound"], v["upper_bound"]),
            is_objective=v["is_objective"],
            maximize=maximize,
        )

    if len(domain.output_variables) < 2:
        logging.debug("Only multivariable optimization is supported currently")
        return

    # Set up models
    num_outputs = len(domain.output_variables)
    num_inputs = domain.num_variables()
    models = {}
    for v in domain.variables:
        if v.is_objective:
            models[str(v.name)] = GPyModel(input_dim=num_inputs)

    num_experiments = int(num_experiments)
    if len(experiments) == 0:
        # Build an initial design
        ld = LatinDesigner(domain)
        results = ld.generate_experiments(num_experiments)
        results = results.to_frame()
        new_results_list = results.to_dict(orient="records")
    else:
        # Get experimental data into a dataset
        experiment_data = [experiment["data"] for experiment in experiments]
        df = pd.DataFrame(experiment_data)
        for v in variables:
            v_id = v.get_id()
            # Find the L2 norm for target values
            if v["is_objective"] and v["objective_type"] == "target":
                df[v_id] = (df[v_id] - v["target_value"]) ** 2
        data = DataSet.from_df(df)

        logging.info(f"Running optimization for project {project_id}")
        # Run the optimization
        try:
            tsemo = TSEMOs(domain=domain, models=models)
            results = tsemo.generate_experiments(
                previous_results=data, num_experiments=num_experiments
            )

            for v in variables:
                v_id = v.get_id()
                # Scale values again
                if v["is_objective"] and v["objective_type"] == "target":
                    results[v_id].where(results[v_id] > 0, 0, inplace=True)
                    results[v_id] = results[v_id] ** (1 / 2) + v["target_value"]
        except Exception as e:
            logging.error(
                "Error encountered when generating experiments for {project_id}: {e}"
            )
            return

        logging.info(f"Returning result for project {project_id}")
        # Convert from dataset to json list
        results_list = results.to_dict(orient="records")
        new_results_list = []
        for result in results_list:
            new_results_dict = {}
            for key, value in result.items():
                new_results_dict[key[0]] = value
            new_results_list.append(new_results_dict)
        new_results_list

    return new_results_list


if __name__ == "__main__":
    anvil_key = os.getenv("ANVIL_UPLINK_KEY")
    print("Connecting to Anvil")
    server.connect(anvil_key)
    print("Starting server")
    try:
        server.wait_forever()
    except KeyboardInterrupt:
        print("Stopping Server")
