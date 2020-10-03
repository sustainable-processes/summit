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

To get started, follow our tutorial_. You can find a more detailed treatment of Summit in our preprint_. 

Also, give us a ⭐ on Github_!

.. _tutorial : tutorial.ipynb
.. _preprint : https://chemrxiv.org/articles/preprint/Summit_Benchmarking_Machine_Learning_Methods_for_Reaction_Optimisation/12939806
.. _Github : https://github.com/sustainable-processes/summit

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   installation
   tutorial
   domains
   experiments_benchmarks/index
   strategies
   transforms
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
