#!/bin/bash

docker run --runtime=nvidia \
	-v $PWD:$PWD \
	-v /allen/aics/modeling/gregj/projects/pytorch_integrated_cell:/root/projects/pytorch_integrated_cell \
	-v /raid/shared:/raid/shared \
	aics/pytorch_integrated_cell \
	/bin/bash -c " cd $PWD; bash run_3D.sh '$1' $2 $3 $4 $5"
