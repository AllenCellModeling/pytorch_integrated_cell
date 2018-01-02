cd ..
python /root/projects/pytorch_integrated_cell/train_model.py \
	--lrEnc 2E-4 --lrDec 2E-4 --lrEncD 1E-2 --lrDecD 2E-4 \
	--encDRatio 1E-4 --decDRatio 1E-5 \
	--model_name aaegan3Dv6-exp \
	--save_dir ./test_aaegan/dgx_test_v4_1/ \
	--train_module aaegan_trainv6 \
	--noise 1E-2 \
	--imdir /root/data/ipp/ipp_17_10_25 \
	--dataProvider DataProvider3Dh5 \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 2 --channels_pt2 0 1 2 \
	--gpu_ids 4 5 \
	--batch_size 30  \
	--nlatentdim 128 \
	--nepochs 150 \
	--nepochs_pt2 150
