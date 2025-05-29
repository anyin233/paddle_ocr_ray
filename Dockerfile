FROM continuumio/miniconda3 as build

COPY paddle.tar.gz .

RUN mkdir -p /opt/conda/env && \
    tar -xzf paddle.tar.gz -C /opt/conda/env && \
    rm paddle.tar.gz

RUN /opt/conda/env/bin/conda-unpack

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirror.katapult.io/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirror.katapult.io/ubuntu/|g' /etc/apt/sources.list

    
RUN apt-get update && \
    apt install python3.10 python-pip build-essential cmake gcc git python3-opencv -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/conda/env /opt/conda/env

WORKDIR /app

COPY . .

# CMD ["serve", "run", "config.yaml"]
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /opt/conda/env/bin/activate && \
    serve run config.yaml
