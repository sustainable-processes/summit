Welcome to Summit's documentation!
==================================

.. image:: _static/banner_4.png
  :alt: Summit banner

Summit is a set of tools for optimising chemical processes. We’ve started by targeting reactions.

What is Summit?
##################
Currently, reaction optimisation in the fine chemicals industry is done by intuition or design of experiments. Both scale poorly with the complexity of the problem. 

Summit uses recent advances in machine learning to make the process of reaction optimisation faster. Essentially, it applies algorithms that learn which conditions (e.g., temperature, stoichiometry, etc.) are important to maximising one or more objectives (e.g., yield, enantiomeric excess). This is achieved through an iterative cycle.

Summit has two key features:

* **Strategies**: Optimisation algorithms designed to find the best conditions with the least number of iterations. Summit has eight strategies implemented.
* **Benchmarks**: Simulations of chemical reactions that can be used to test strategies. We have both mechanistic and data-driven benchmarks.

We suggest trying one of our tutorials_ or reading our publication_ (or preprint_). Also, give us a ⭐ on Github_!

Below is a quick start that demonstrates the functionality of Summit:

.. code-block:: python

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

.. _tutorials : tutorials/index.rst
.. _publication : https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cmtd.202000051
.. _preprint : https://chemrxiv.org/articles/preprint/Summit_Benchmarking_Machine_Learning_Methods_for_Reaction_Optimisation/12939806
.. _Github : https://github.com/sustainable-processes/summit

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   installation
   tutorials/index
   domains
   experiments_benchmarks/index
   strategies
   runner
   transforms
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
