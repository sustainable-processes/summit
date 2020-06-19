.. role:: raw-html-m2r(raw)
   :format: html


Summit
======

Summit is a set of tools for optimizing chemical processes. 

Installation
------------

If you want to use summit immediately without installing python on your computer, go to our `Jupyterhub <hub.rxns.io>`_\ , which already has it installed. You can find a description of Jupyterhub `here <https://github.com/sustainable-processes/server/blob/master/notes/session_1.md>`_.

To install locally:

``pip install git+https://github.com/sustainable-processes/summit_private.git@0.3.0#egg=summit``

You might need to enter your username and password for Github. 

Documentation
-------------

The documentation for summit can be found on the `wiki <https://github.com/sustainable-processes/summit/wiki>`_.
:raw-html-m2r:`<!-- It would be great to add a "Quick Start" here.-->`

Development
-----------

Downloading the code
^^^^^^^^^^^^^^^^^^^^


#. Clone the repository:
   ``git clone https://github.com/sustainable-processes/summit_private.git``
#. Intall poetry by following the instructions `here <https://python-poetry.org/docs/#installation>`_. We use poetry for dependency management.
#. Install all dependencies:
   ``poetry install``
#. To run tests:
   ``poetry run pytest --doctest-modules --ignore=case_studies``

Commit Worfklow
^^^^^^^^^^^^^^^


* Use the `project board <https://github.com/orgs/sustainable-processes/projects/1>`_ to keep track of issues. Issues will automatically be moved along in the board when they are closed in Github.
* Write tests in the tests/ folder
* Documentation follows the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html#documenting-class-instances>`_

  * Please include examples when possible that can be tested using `doctest <https://docs.python.org/3/library/doctest.html>`_
  * All publicly available classes and methods should have a docstring

* Commit to a branch off master and submit pull requests to merge. 

  * To create a branch locally and push it:
    .. code-block:: bash

       $ git checkout -b BRANCH_NAME
       # Once you've made some changes
       $ git commit -am "commit message"
       $ git push -u origin BRANCH_NAME
       #Now if you come back to Github, your branch should exist

  * All pull requests need one review.
  * Tests will be run automatically when a pull request is created, and all tests need to pass before the pull request is merged. 

Releases
^^^^^^^^

Below is the old process for building a release. In the future, we will have this automated using Github actions.


#. Install `s3pypi <https://github.com/novemberfiveco/s3pypi>`_ and `dephell <https://dephell.org/docs/installation.html>`_
#. Install AWS credentials to upload pypi.rxns.io (Kobi is the one who controls this).
#. Bump the version in pyproject.toml and then run:
    ``dephell deps convert --from=pyproject.toml --to=setup.py``
#. Go into setup.py and delete the lines for extras_install_requires
#. Upload the package to the private pypi repository:
    ``s3pypi --bucket pypi.rxns.io``
