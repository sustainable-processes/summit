.. Summit documentation master file, created by
   sphinx-quickstart on Fri Aug 28 21:25:06 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Summit's documentation!
==================================

Summit is a set of tools for optimising chemical processes. We've started by targeting reactions.

.. Put a gif here

Currently, reaction optimisation in the fine chemicals industry is done by intuition or design of experiments, which both scale poorly with the complexity of the problem. Summit applies recent advances in machine learning to make the process of reaction optimisation faster. Essentially, it applies algorithms that learn which conditions (e.g., temperature, stoichiometry, etc.) are important to maximising one or more objectives (e.g., yield, enantiomeric excess). This is achieved through an iterative cycle.

For a more academic treatment of Summit, check out "Benchmarking Machine Learning for Reaction Optimisation." If you just want to try it, out, check out our tutorial.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   tutorial
   new_benchmarks


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
