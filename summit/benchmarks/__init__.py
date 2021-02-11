from .snar import SnarBenchmark
from .test_functions import Himmelblau, Hartmann3D, ThreeHumpCamel, DTLZ2, VLMOP2

from .experimental_emulator import (
    ExperimentalEmulator,
    ReizmanSuzukiEmulator,
    registry,
    ANNRegressor,
    BNNRegressor,
    CrossValidate,
    # BaumgartnerCrossCouplingEmulator,
    # BaumgartnerCrossCouplingDescriptorEmulator,
    # BaumgartnerCrossCouplingEmulator_Yield_Cost,
)
from .MIT import *

# Register regressors here
registry.register(ANNRegressor)
registry.register(BNNRegressor)
