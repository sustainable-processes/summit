import pathlib
import shutil


def get_summit_config_path(config_dir_name=".summit"):
    """Returns the path to the summit config directory"""
    home = pathlib.Path.home()
    return home / config_dir_name


def clean_house(config_dir_name=".summit"):
    """This removes all the temporary files stored in the config directory"""
    config_path = get_summit_config_path(config_dir_name)
    shutil.rmtree(config_path)


from summit.domain import *
from summit.experiment import *
from summit.run import *
from summit.strategies import *
from summit.benchmarks import *
from summit.utils.dataset import DataSet
from summit.utils.multiobjective import pareto_efficient, hypervolume


def run_tests():
    """Run tests using pytest"""
    import pytest
    from pytest import ExitCode
    import sys

    retcode = pytest.main(
        ["--doctest-modules", "--disable-warnings", "--pyargs", "summit"]
    )
    sys.exit(retcode)