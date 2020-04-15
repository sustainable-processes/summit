
Summit
======

Summit is a set of tools for optimizing chemical processes. 

Installation
------------

If you want to use summit immediately without installing python on your computer, go to our `Jupyterhub <hub.rxns.io>`_\ , which already has it installed. You can find a description of it `here <https://github.com/sustainable-processes/server/blob/master/notes/session_1.md>`_.

To install locally:

``pip install git+https://github.com/sustainable-processes/summit.git@0.2.2#egg=summit``

You might need to enter your username and password for Github. 

Documentation
-------------

The documentation for summit can be found on the `wiki <https://github.com/sustainable-processes/summit/wiki>`_.

Case Studies
------------

In addition to the documentation, we are prepareing several case studies.  These contain jupyter notebooks with practical examples to follow. 


* `Formulation <case_studies/formulation>`_
* `Nanosilica <case_studies/nanosilica>`_
* `Photo Amination <case_studies/photoamination/>`_
* `Borrowing Hydrogen <case_studies/borrowing_hydrogen>`_

Develpment
----------

Build a release
^^^^^^^^^^^^^^^


#. Install `s3pypi <https://github.com/novemberfiveco/s3pypi>`_ and `dephell <https://dephell.org/docs/installation.html>`_
#. Install AWS credentials to upload pypi.rxns.io (Kobi is the one who controls this).
#. Bump the version in pyproject.toml and then run:
    ``dephell deps convert --from=pyproject.toml --to=setup.py``
#. Go into setup.py and delete the lines for extras_install_requires
#. Upload the package to the private pypi repository:
    ``s3pypi --bucket pypi.rxns.io``

Building Containers
^^^^^^^^^^^^^^^^^^^

This is useful when you want to run summit on a remote server. 


#. 
   Download `habitus <https://www.habitus.io/>`_. Habitus is used to insert ssh keys into containers safely, so you can pull from for private Github repositories. 

#. 
   If you are on mac, you need to install gnu-tar:

   .. code-block::

       brew install gnu-tar
       echo "export PATH=/usr/local/opt/gnu-tar/libexec/gnubin:$PATH" >> ~./bash_profile
       echo "aliast tar='gtar'" >> ~/.bash_profile
       source ~/.bash_profile

#. 
   Change line 10 of the build.yml to be the name of your `ssh key for github <https://help.github.com/en/articles/connecting-to-github-with-ssh>`_ 

#. 
   Inside the main folder for summit (i.e., the same folder as this README), build the container:

   .. code-block::

       #On Mac
       sudo habitus --build host=docker.for.mac.localhost --binding=127.0.0.1 --secrets=true

       #This is what I think it will be on linux
       sudo habitus --build host=gateway.docker.local --binding=127.0.0.1 --secrets=true
