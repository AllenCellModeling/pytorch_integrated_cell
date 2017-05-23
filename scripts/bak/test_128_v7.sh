cd ..
python train_model.py --gpu_ids 0 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --nepochs_pt2 500 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 2E-4 --lrDecD 2E-4 --encDRatio 1E-3 --decDRatio 1E-5 --model_name aaegan_256v2 --save_dir ./test_aaegan/aaegan_128_v7/ --train_module aaegan_train --noise=0.01
