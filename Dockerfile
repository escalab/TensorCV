FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    vim build-essential git cmake python3 python3-pip\
    && rm -rf /var/lib/apt/lists/*

VOLUME /workspace