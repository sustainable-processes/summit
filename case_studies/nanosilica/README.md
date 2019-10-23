# Nanosilica Demo

This is a demo of using summit to optimize conditions for nanomaterial synthesis.  It is a multiobjective optimization with constraints and several manipulated variables.

## Installation
This requires [python 3.6](https://www.python.org/downloads/release/python-360/) or higher to be installed.

1. **Download** the Demos zip file from the [releases page](https://github.com/sustainable-processes/summit/releases) .  
2. **Unzip** the file. Then, navigate to the nanosilica folder using the command line.
3. **Run these commands** inside the nanosilica folder:
    ```bash
    pip3 install venv --upgrade
    python3 -m venv nanosilica-venv
    source formulation-venv/bin/activate
    pip install summit jupyterlab --extra-index-url https://pypi.rxns.io
    jupyter lab
    ```
4. **Run the formulation demo**: [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/)** tab will appear in your browser. Double click on `formulation.ipynb` and from the menu click "Run > Run All". 