FROM tensorflow/tensorflow:2.0.0-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      screen \
      wget && \
    rm -rf /var/lib/apt/lists/* \
    apt-get upgrade

WORKDIR /ISR

RUN pip install --upgrade pip

RUN pip install ISR numpy pillow 