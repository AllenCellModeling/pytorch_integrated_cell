cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --nepochs_pt2 250 --lrEnc 1E-4 --lrDec 1E-4 --lrEncD 1E-4 --lrDecD 1E-4 --encDRatio 1 --decDRatio 1 --model_name waaegan_v3 --save_dir ./test_waaegan/waaegan_improved/ --train_module waaegan_train --DitersAlt 20 --improved True
