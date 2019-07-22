from integrated_cell.utils.build_control_images import build_control_images

save_dir = "/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/controls"
csv_path = "/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/data_jobs_out.csv"
image_parent = "/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/"

build_control_images(save_dir, csv_path, image_parent)
