from summit import *
import matplotlib.pyplot as plt
import numpy as np

baumgartner = get_pretrained_baumgartner_cc_emulator()
levels_dict = {
    "catalyst": 2,
    "base": 2,
    "t_res": 1,
    "temperature": 1,
    "base_equivalents": 2,
}

strategy = CBBO(domain=baumgartner.domain, levels_dict=levels_dict)

baumgartner.reset()
r = Runner(
    strategy=strategy,
    experiment=baumgartner,
    batch_size=1,
    max_iterations=3,
)
r.run()
