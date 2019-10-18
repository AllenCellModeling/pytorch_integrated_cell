#!/bin/bash

docker run -it --runtime=nvidia \
	-v $PWD/../../:$PWD/../../ \
	-v $PWD/../../:/root/projects/pytorch_integrated_cell \
	-v /raid/shared:/raid/shared \
	aics/pytorch_integrated_cell \
	/bin/bash
