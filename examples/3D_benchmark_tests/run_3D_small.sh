gpu_ids=$1
save_dir=$2
opt_level=$3
batch_size=16

# image_dir=../../data/
# csv_name=metadata.csv

image_dir=/raid/shared/ipp/scp_19_04_10/
csv_name=controls/data_plus_controls.csv

ic_train_model \
	--gpu_ids $gpu_ids \
	--model_type ae \
	--save_dir $save_dir \
	--lr_enc 2E-4 --lr_dec 2E-4 \
	--data_save_path $save_dir/data.pyt \
	--crit_recon torch.nn.MSELoss \
	--kwargs_crit_recon '{"reduction": "sum"}' \
	--network_name vaegan3D_cgan \
	--kwargs_enc '{"n_classes": 24, "n_channels": 2, "n_channels_target": 1, "n_latent_dim": 512, "n_ref": 512, "conv_channels_list": [32, 64, 128]}'  \
    --kwargs_enc_optim '{"betas": [0.9, 0.999]}' \
    --kwargs_dec '{"n_classes": 24, "n_channels": 2, "n_channels_target": 1, "n_latent_dim": 512, "n_ref": 512, "proj_z": 0, "proj_z_ref_to_target": 0, "activation_last": "softplus", "conv_channels_list": [128, 64, 32]}' \
	--kwargs_dec_optim '{"betas": [0.9, 0.999]}' \
	--kwargs_model '{"beta": 1, "objective": "H", "save_state_iter": 1, "save_progress_iter": 1, "opt_level": "'$opt_level'"}' \
	--train_module cbvae_apex \
	--imdir $image_dir \
	--dataProvider DataProvider \
	--kwargs_dp '{"rescale_to": 0.25, "crop_to": [40, 24, 16], "check_files": 0, "csv_name": "'$csv_name'"}' \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 1 2 \
	--batch_size $batch_size  \
	--nepochs 5 \
