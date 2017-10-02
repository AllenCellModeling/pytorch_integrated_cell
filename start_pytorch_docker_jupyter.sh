#!/usr/bin/env bash

PORT=${1}
PROJ_DIR=$(dirname $(pwd -P))
DATA_DIR='/allen/aics/modeling/data'
RSLT_DIR='/allen/aics/modeling/gregj/results'

nvidia-docker run -it \
    -e "PASSWORD=jupyter1" \
    -p ${PORT}:9999 \
    -v ${PROJ_DIR}:/root/projects \
    -v ${DATA_DIR}:/root/data \
    -v ${RSLT_DIR}:/root/results \
    rorydm/pytorch_extended:pytorch0.2_conda \
    bash -c "jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000000"

