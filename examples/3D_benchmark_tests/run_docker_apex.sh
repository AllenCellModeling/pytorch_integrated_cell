#!/bin/bash

docker run --detach --runtime=nvidia \
	-v $PWD/../../:$PWD/../../ \
	-v $PWD/../../:/root/projects/pytorch_integrated_cell \
	-v /raid/shared:/raid/shared \
	aics/pytorch_integrated_cell \
	/bin/bash -c " cd $PWD; bash $1 '$2' $3 '$4'"
