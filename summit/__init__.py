import os
import pathlib

def get_summit_config_path(config_dir_name=".summit"):
    home = pathlib.Path.home()
    return home / config_dir_name


from summit.domain import *
from summit.experiment import Experiment
from summit.run import Runner, NeptuneRunner
from summit.strategies import *
