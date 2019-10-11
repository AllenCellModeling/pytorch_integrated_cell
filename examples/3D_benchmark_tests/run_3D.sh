gpu_ids=$1
save_dir=$2
trainer_type=$3
batch_size=$4
ndat=$5

ic_train_model \
	--gpu_ids $gpu_ids \
	--model_type ae \
	--save_dir $save_dir \
	--lr_enc 2E-4 --lr_dec 2E-4 \
	--data_save_path data.pyt \
	--crit_recon torch.nn.MSELoss \
	--kwargs_crit_recon '{"reduction": "sum"}' \
	--network_name vaegan3D_cgan \
	--kwargs_enc '{"n_classes": 24, "n_channels": 2, "n_channels_target": 1, "n_latent_dim": 512, "n_ref": 512}'  \
        --kwargs_enc_optim '{"betas": [0.9, 0.999]}' \
        --kwargs_dec '{"n_classes": 24, "n_channels": 2, "n_channels_target": 1, "n_latent_dim": 512, "n_ref": 512, "proj_z": 0, "proj_z_ref_to_target": 0, "activation_last": "softplus"}' \
	--kwargs_dec_optim '{"betas": [0.9, 0.999]}' \
	--kwargs_model '{"beta": 1, "objective": "H", "save_state_iter": 1E9, "save_progress_iter": 1E9}' \
	--train_module $trainer_type \
	--imdir /raid/shared/ipp/scp_19_04_10/ \
	--dataProvider DataProvider \
	--kwargs_dp '{"crop_to": [160,96, 64], "return2D": 0, "check_files": 1, "make_controls": 0, "csv_name": "controls/data_plus_controls.csv"}' \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 1 2 \
	--batch_size $batch_size  \
	--nepochs 1 \
	--ndat $ndat
