FROM jonatkinson/python-poetry:3.6

#Docker container to be run with habitus build tool (https://www.habitus.io/)

ARG host
RUN git config --global url."git@github.com:".insteadOf "https://github.com/"
RUN mkdir ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
    # echo "Host *\n\tAddKeysToAgent yes\n\tUseKeychain yes\n\tIdentityFile ~/.ssh/id_rsa" >> ~/.ssh/config

#Run poetry install with my ssh secrets installed and then removed
COPY summit/ pyproject.toml poetry.lock /
# RUN wget -O ~/.ssh/id_rsa http://$host:8080/v1/secrets/file/id_rsa && chmod 0600 ~/.ssh/id_rsa  && \
#     eval "$(ssh-agent -s)"  && ssh-add ~/.ssh/id_rsa && \
#     poetry install && \
#     rm ~/.ssh/id_rsa

