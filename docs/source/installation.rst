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

    # with poetryff
    poetry add summit -E entmoot

    # with pipenv
    pipenv install summit[entmoot]


Notes about installing on Apple M1
***********************************

You might run into some issues when installing scientific python packages such as Summit on Apple M1. Follow the steps below to install via pip:

.. code-block:: bash
    arch -arm64 brew install llvm@11 
    brew install hdf5
    HDF5_DIR=/opt/homebrew/opt/hdf5 PIP_NO_BINARY="h5py" LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" arch -arm64 poetry install

More resources

* [LLVM isssue](https://github.com/numba/llvmlite/issues/693#issuecomment-909501195)
* [Installing H5py (for numpy)](https://docs.h5py.org/en/stable/build.html#custom-installation)
