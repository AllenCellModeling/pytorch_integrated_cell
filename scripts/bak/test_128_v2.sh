cd ..
python train_model.py --gpu_ids 0 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 2E-4 --lrDecD 2E-4 --encDRatio 1E-3 --decDRatio 1E-4 --model_name aaegan_256 --save_dir ./test_aaegan/aaegan_128_v2/ --train_module aaegan_train --noise=0.01
