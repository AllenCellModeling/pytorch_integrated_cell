cd ..
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 250 --nepochs_pt2 250 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 2E-4 --lrDecD 2E-4 --encDRatio 1E-3 --decDRatio 1E-5 --model_name waaegan_256v2 --save_dir ./test_waaegan/waaegan_128/ --train_module waaegan_train --DitersAlt 20
