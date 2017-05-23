cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --lrEnc 5E-5 --lrDec 5E-5 --lrEncD 5E-5 --lrDecD 5E-5 --encDRatio 1E-3 --decDRatio 5E-5 --model_name aaegan_256 --save_dir ./test_aaegan/aaegan_128/ --train_module aaegan_train --noise=0.01
