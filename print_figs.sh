MODEL_DIR='/root/results/integrated_cell/test_aaegan/dgx_test_v4_1/'
GPU_IDS='0 1 2 3'


#python print_latent_walk_directed.py --parent_dir $MODEL_DIR --gpu_ids $GPU_IDS

#python print_images.py --parent_dir $MODEL_DIR --gpu_ids $GPU_IDS

#python print_variation_data.py --parent_dir $MODEL_DIR --gpu_ids $GPU_IDS
#python print_variation_model_structure.py --parent_dir $MODEL_DIR --gpu_ids $GPU_IDS
python print_variation_model_structure_sampled.py --parent_dir $MODEL_DIR --gpu_ids $GPU_IDS
python print_variation_figures.py --parent_dir $MODEL_DIR

