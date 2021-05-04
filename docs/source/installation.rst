Installation
============

The easiest way to install Summit is using pip or a depedency manager that supports pip:


.. code-block:: bash

    pip install summit

You could also use poetry or pipenv:

.. code-block:: bash

    poetry add summit

.. code-block:: bash

    pipenv install summit


Summit has a set of extra dependencies for running the code in the experiments folder on Github_. You can install them as follows:

.. code-block:: bash

    # with pip:
    pip install summit[experiments]

    # with poetry
    poetry add summit -E experiments

    # with pipenv
    pipenv install summit[experiments]


.. _Github: https://github.com/sustainable-processes/summit/tree/master/experiments

Additionally, if you want to use the experimental ENTMOOT feature you need to install the ENTMOOT package

.. code-block:: bash

    # with pip:
    pip install summit[entmoot]

    # with poetry
    poetry add summit -E entmoot

    # with pipenv
    pipenv install summit[entmoot]
