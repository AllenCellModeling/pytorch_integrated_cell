ic_train_model \
        --gpu_ids $1 \
        --model_type ae \
        --save_dir $PWD/bvae3D \
        --lr_enc 2E-4 --lr_dec 2E-4 \
        --data_save_path $PWD/bvae3D/data_ae.pyt \
        --crit_recon integrated_cell.losses.BatchMSELoss \
        --kwargs_crit_recon '{}' \
        --network_name cvaegan3D_residual \
        --kwargs_model '{"beta_min": 1e-06, "beta_start": -1, "beta_step": 3e-05, "kld_reduction": "batch", "objective": "A"}' \
        --kwargs_enc '{"n_latent_dim": 512, "n_ch_target": 3, "n_ch_ref": 0, "n_classes": 0}'  \
        --kwargs_enc_optim '{"betas": [0.9, 0.999]}' \
        --kwargs_dec '{"n_latent_dim": 512, "activation_last": "softplus", "n_ch_target": 3, "n_ch_ref": 0, "n_classes": 0}' \
        --kwargs_dec_optim '{"betas": [0.9, 0.999]}' \
        --kwargs_model '{"beta": 0.01}' \
        --train_module bvae \
        --imdir $PWD/../../data/ \
        --dataProvider DataProvider \
        --kwargs_dp '{"crop_to": [160, 96, 64], "return2D": 0, "check_files": 0, "csv_name": "metadata.csv"}' \
        --saveStateIter 1 --saveProgressIter 1 \
        --channels_pt1 0 1 2 \
        --batch_size 32  \
        --nepochs 300 \
