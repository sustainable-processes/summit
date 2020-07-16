FROM python:3.7

WORKDIR /summit_user
COPY setup.py ./
COPY summit summit/
RUN ls && pip install .
ENTRYPOINT ["python"] 

