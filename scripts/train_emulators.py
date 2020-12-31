from summit.benchmarks import ReizmanSuzukiEmulator
from summit.benchmarks.experimental_emulator import ANNRegressor
from summit.utils.dataset import DataSet

import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = DataSet.read_csv("data/reizman_suzuki_case_1.csv")
    exp = ReizmanSuzukiEmulator(case=1, dataset=ds, regressor=ANNRegressor)
    exp.train(max_epochs=10)

    exp = ReizmanSuzukiEmulator(case=1, load_checkpoint=True, regressor=ANNRegressor)
    exp.pareto_plot()
    plt.show()
