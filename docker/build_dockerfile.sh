#!/usr/bin/env bash

cd ..
docker build -t aics/pytorch_integrated_cell -f docker/Dockerfile .
