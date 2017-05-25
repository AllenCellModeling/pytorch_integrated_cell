cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --nepochs_pt2 250 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 2E-4 --lrDecD 2E-4 --encDRatio 1E-3 --decDRatio 1E-5 --model_name aaegan_v2 --save_dir ./test_aaegan/aaegan_v2_v1/ --train_module aaegan_train
