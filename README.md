# Summit
![summit_banner](https://raw.githubusercontent.com/sustainable-processes/summit/master/docs/source/_static/banner_4.png)

<p align="center">
<a href='https://gosummit.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/gosummit/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://pypi.org/project/nsummit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/summit"></a>
</p>

Summit is a set of tools for optimising chemical processes. We’ve started by targeting reactions.

## What is Summit?
Currently, reaction optimisation in the fine chemicals industry is done by intuition or design of experiments.  Both scale poorly with the complexity of the problem. 

Summit uses recent advances in machine learning to make the process of reaction optimisation faster. Essentially, it applies algorithms that learn which conditions (e.g., temperature, stoichiometry, etc.) are important to maximising one or more objectives (e.g., yield, enantiomeric excess). This is achieved through an iterative cycle.

Summit has two key features:

- **Strategies**: Optimisation algorithms designed to find the best conditions with the least number of iterations. Summit has eight strategies implemented.
- **Benchmarks**: Simulations of chemical reactions that can be used to test strategies. We have both mechanistic and data-driven benchmarks.

To get started, see the Quick Start below or follow our [tutorial](https://gosummit.readthedocs.io/en/latest/tutorial.html). 

## Installation

To install summit, use the following command:

```pip install summit```

## Quick Start

Below, we show how to use the Nelder-Mead  strategy to optimise a benchmark representing a nucleophlic aromatic substitution (SnAr) reaction.
```python
# Import summit
from summit.benchmarks import SnarBenchmark
from summit.strategies import NelderMead, MultitoSingleObjective
from summit.run import Runner

# Instantiate the benchmark
exp = SnarBenchmark()

# Since the Snar benchmark has two objectives and Nelder-Mead is single objective, we need a multi-to-single objective transform
transform = MultitoSingleObjective(
    exp.domain, expression="-sty/1e4+e_factor/100", maximize=False
)

# Set up the strategy, passing in the optimisation domain and transform
nm = NelderMead(exp.domain, transform=transform)

# Use the runner to run closed loop experiments
r = Runner(
    strategy=nm, experiment=exp,max_iterations=50
)
r.run()
```

## Documentation

The documentation for summit can be found [here](https://gosummit.readthedocs.io/en/latest/index.html).


## Issues?
Submit an [issue](https://github.com/sustainable-processes/summit/issues) or send an email to kcmf2@cam.ac.uk.

## Citing

If you find this project useful, we encourage you to

* Star this repository :star: 
* Cite our [paper](https://chemrxiv.org/articles/preprint/Summit_Benchmarking_Machine_Learning_Methods_for_Reaction_Optimisation/12939806).
```
@article{Felton2020,
author = "Kobi Felton and Jan Rittig and Alexei Lapkin",
title = "{Summit: Benchmarking Machine Learning Methods for Reaction Optimisation}",
year = "2020",
month = "9",
url = "https://chemrxiv.org/articles/preprint/Summit_Benchmarking_Machine_Learning_Methods_for_Reaction_Optimisation/12939806",
doi = "10.26434/chemrxiv.12939806.v1"
}
```

