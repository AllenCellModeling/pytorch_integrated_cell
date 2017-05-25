cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --nepochs_pt2 250 --lrEnc 5E-5 --lrDec 5E-5 --lrEncD 5E-5 --lrDecD 5E-5 --encDRatio 1 --decDRatio 1 --model_name waaegan_v3 --save_dir ./test_waaegan/waaegan_v3/ --train_module waaegan_train --DitersAlt 20
