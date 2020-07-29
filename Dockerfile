FROM python:3.7

WORKDIR /summit_user
COPY requirements.txt ./
# Have to install numpy first due to Gryffin
RUN  pip install numpy==1.18.0 && pip install -r requirements.txt
COPY setup.py ./
COPY summit summit/
RUN pip install .
ENTRYPOINT ["python"] 

