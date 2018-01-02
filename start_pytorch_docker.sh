nvidia-docker run -it \
	-v /allen/aics/modeling/gregj/projects:/root/projects \
        -v /allen/aics/modeling/gregj/results:/root/results \
        -v /raid/gregj/cache:/root/data \
	gregj/pytorch_extras:dgx_jupyter \
	bash 
