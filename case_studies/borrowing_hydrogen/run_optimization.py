import papermill as pm
from summit.data import DataSet


previous_results = DataSet.read_csv(experimental_datafile)
num_batches = previous_results["batch"].max()

pm.execute_notebook(
    "optimization_template.ipynb",
    f"outputs/optimization_batch_{i}.ipynb",
    parameters=dict(alpha=0.6, ratio=0.1),
)
