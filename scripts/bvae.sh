cd ..
python /root/projects/pytorch_integrated_cell/train_model.py \
	--gpu_ids $1 \
	--save_parent ./results/test_bvae/ \
	--lrEnc 1E-4 --lrDec 1E-4 \
	--data_save_path ./results/data.pyt \
	--critRecon BCELoss \
	--model_name vaaegan3D \
	--kwargs_model '{"beta": 1}' \
	--train_module bvae \
	--kwargs_optim '{"betas": [0.9, 0.999]}' \
	--imdir /root/results/ipp/ipp_17_10_25 \
	--dataProvider DataProvider3Dh5 \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 2 --channels_pt2 0 1 2 \
	--batch_size 16  \
	--nlatentdim 128 \
	--nepochs 50 \
	--nepochs_pt2 50 \
