nvidia-docker run -it \
	-e "PASSWORD=jupyter1" \
	-p 9602:9999 \
	-v /allen/aics/modeling/rorydm/pytorch_integrated_cell:/root/projects \
	-v /allen/aics/modeling/data:/root/data \
	gregj/pytorch_extended \
	bash -c "jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000000"

