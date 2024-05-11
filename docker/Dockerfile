FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

## User environment
#
ENV DEV_USER=devusr
ENV DEV_GROUP=devgrp
ENV DEV_UID=1110
ENV DEV_GID=1110
ENV APP_HOME=/app
ENV PYTHON_VER_MAJ=3.11
ENV PYTHON_VER_MIN=${PYTHON_VER_MAJ}.8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


## Build tools
#
# install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        file \
        wget \
        vim \
        graphviz \
        build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm \
        libncurses5-dev libncursesw5-dev xz-utils tk-dev \
        libffi-dev


# install Python
#
WORKDIR /tmp
RUN wget --no-verbose https://www.python.org/ftp/python/${PYTHON_VER_MIN}/Python-${PYTHON_VER_MIN}.tgz
RUN tar xf Python-${PYTHON_VER_MIN}.tgz
WORKDIR Python-${PYTHON_VER_MIN}
RUN ./configure --enable-optimizations && \
	make -j8 && \
	make altinstall
RUN cd /bin && \
    ln -s /usr/local/bin/python${PYTHON_VER_MAJ} python && \
    ln -s /usr/local/bin/pip${PYTHON_VER_MAJ} pip


# install packages
#
RUN mkdir /tmp/packages
WORKDIR /tmp/packages
COPY target/packages/* ./
RUN pip install *.whl
RUN python -m spacy download en_core_web_sm


## Setup and install user account
#
# create development user/group
RUN addgroup --gid $DEV_GID $DEV_GROUP && \
    adduser --gecos 'Development User' --disabled-password \
      --uid $DEV_UID --gid $DEV_GID --home ${APP_HOME} $DEV_USER


# setup application
#
RUN mkdir -p ${APP_HOME}/data && \
    mkdir -p ${APP_HOME}/model && \
    mkdir -p ${APP_HOME}/Desktop/amr-graph
WORKDIR ${APP_HOME}
COPY target/app .
RUN ln -s ${APP_HOME}/app.config .calamrrc
RUN ln -s ${APP_HOME}/model .cache
RUN ln -s ${APP_HOME}/Desktop/amr-graph ${APP_HOME}/results
RUN chown -R ${DEV_USER}:${DEV_GROUP} .


## Cleanup
#
RUN apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/*


## Create the adhoc micro corpus
#
RUN su - devusr --command 'calamr mkadhoc --corpusfile corpus/micro/source.json'
