
Summit
======

Summit is a set of tools for optimizing chemical reactions. 

Case Studies
------------


* `Photo Amination <case_studies/photoamination/>`_
* `Borrowing Hydrogen <case_studies/borrowing_hydrogen>`_

Building Containers
-------------------


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
