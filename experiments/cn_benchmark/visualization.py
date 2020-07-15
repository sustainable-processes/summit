from cn_benchmark_visualization import PlotExperiments
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import pandas as pd

pe = PlotExperiments('sustainable-processes/summit', experiment_ids=[f'SUM-{id}' for id in range(1137,1149)])

iterations_table = pe.iterations_to_threshold(yld_threshold=0.95)
iterations_table.to_csv('data/.csv')
iterations_table