cd ..
python train_model.py --gpu_ids 0 1 2 --batch_size 30 --data_save_path ./test_aaegan/aaegan3Dv8_v1-exp/data.pyt --nlatentdim 128 --nepochs 100 --nepochs_pt2 125 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 1E-2 --lrDecD 2E-4 --encDRatio 1E-4 --decDRatio 1E-5 --model_name aaegan3Dv6-exp --save_dir ./test_aaegan/aaegan3Dv8_v1-exp/ --train_module aaegan_trainv6 --noise 1E-3 --imdir /root/results/ipp_17_10_25 --dataProvider DataProvider3Dh5 --saveStateIter 1 --saveProgressIter 1 --channels_pt1 0 2 --channels_pt2 0 1 2
