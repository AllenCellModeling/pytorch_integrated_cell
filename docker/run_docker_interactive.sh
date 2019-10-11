#!/bin/bash

docker run --runtime=nvidia -it \
	aics/pytorch_integrated_cell \
	/bin/bash