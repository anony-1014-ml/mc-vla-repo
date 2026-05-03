# Base
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Install needed apt packages
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore

USER root

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    gnutls-bin \
    openssh-client \
    libghc-x11-dev \
    gcc-multilib \
    g++-multilib \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    xvfb \
    mesa-utils \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    openjdk-8-jdk \
    ffmpeg \
    vim \
    curl \
    software-properties-common \
    grep \
    x11-xserver-utils \
    && apt-get clean

# Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10-dev python3.10-venv && apt-get clean

# venv
RUN python3.10 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"

# Source
RUN mkdir /cab
WORKDIR /cab
COPY . .

# pip
RUN pip install --upgrade pip && \
    pip install -e ./minestudio/external/cab_vla && \
    pip install -e .

# file
RUN chown -R 1000:root .
RUN chmod -R 0777 .
ENV PRISMATIC_MODEL_DIR="/model"

#
RUN python -m minestudio.simulator.entry -y
CMD ["python", "-m", "minestudio.simulator.entry"]
