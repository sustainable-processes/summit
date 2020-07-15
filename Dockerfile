FROM python:3.7

COPY pyproject.toml poetry.lock README.md /
COPY summit/ /summit/
RUN pip install .
ENTRYPOINT python 

