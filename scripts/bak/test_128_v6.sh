cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --lrEnc 5E-4 --lrDec 5E-4 --lrEncD 5E-4 --lrDecD 5E-4 --encDRatio 1E-3 --decDRatio 1E-2 --model_name aaegan_256v3 --save_dir ./test_aaegan/aaegan_128_v6/ --train_module aaegan_train --noise=0.01
