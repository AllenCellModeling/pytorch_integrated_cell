cd ..
rm -r ./test_aaegan/test/
python train_model.py --gpu_ids 1 --batch_size 32 --imsize 128 --nlatentdim 16 --nepochs 2 --lrEnc 5E-5 --lrDec 5E-5 --lrEncD 5E-5 --lrDecD 5E-5 --encDRatio 1E-3 --decDRatio 5E-5 --model_name aaegan_256 --save_dir ./test_aaegan/test/ --train_module aaegan_train --noise=0.01 --ndat 100 --saveProgressIter 1 --saveStateIter 2
