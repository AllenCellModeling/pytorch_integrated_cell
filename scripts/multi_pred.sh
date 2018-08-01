cd ..


python /root/projects/pytorch_integrated_cell/train_model.py \
	--gpu_ids $1 \
	--save_dir ./results/multi_pred \
        --data_save_path ./results/data.pyt \
	--lrEnc 2E-4 --lrDec 2E-4 \
	--model_name multi_pred \
	--train_module multi_pred \
	--kwargs_optim '{"betas": [0.9, 0.999]}' \
	--imdir /root/results/ipp/ipp_17_10_25 \
	--dataProvider DataProvider3Dh5 \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 2 3 4 5 1 --channels_pt2 0 \
	--batch_size 16  \
	--nlatentdim 128 \
	--nepochs 100 \
	--nepochs_pt2 0 \
	--overwrite_opts True \
