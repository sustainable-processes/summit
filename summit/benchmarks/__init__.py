from .snar import SnarBenchmark
from .test_functions import Himmelblau, Hartmann3D, ThreeHumpCamel, DTLZ2, VLMOP2
from .experimental_emulator import *
from .MIT import *


# Create global regressor registry
registry = RegressorRegistry()
registry.register(ANNRegressor)
