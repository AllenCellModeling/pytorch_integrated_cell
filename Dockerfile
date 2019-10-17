# From pytorch compiled from source
FROM nvcr.io/nvidia/pytorch:19.09-py3

COPY ./ /root/projects/pytorch_integrated_cell

WORKDIR /root/projects/pytorch_integrated_cell

RUN pip install -e .
