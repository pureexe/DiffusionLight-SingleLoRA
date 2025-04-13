# load env
bin/diffusionlight_shell

# create chromeball
python validate_2lora.py --dataset /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom1 --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000 --no_save_intermediate --no_torch_compile --no_random_loader --lora_path real_checkpoint/rev3/Flickr2K/Flickr2K_balanced_aligned/checkpoint-140000

# create HDR chromeball
python exposure2hdr.py --input_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000/square --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000/square_hdr

# predict normal (Marigold)
python copy1_compute_normal.py

# predict normal (Lotus)
python infer.py --pretrained_model_name_or_path="jingheya/lotus-normal-g-v1-0" --prediction_type=sample --seed=42 --half_precision --input_dir="/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom1" --task_name="normal" --mode="generation" --output_dir="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000" --processing_res=0

# predict depth and fov
python /ist/ist-share/vision/pakkapon/relight/ml-depth-pro/run_copyroom1.py

# create perspective projection
python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512


# create SHCOEFF
python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100 --total_scene 1 --num_order 100

# create shading
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100 --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_viz_max --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_viz_ldr



######################## RERUN GT 
# create perspective projection
python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_gt --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512

# create SHCOEFF
python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100_gt --total_scene 1 --num_order 100

# create shading
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_viz_ldr_gt

######################################################
python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100 --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6_viz_max --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6_viz_ldr

python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_lotus_exr_order6_viz_ldr_gt

##############################

### orthographic gt 
python ball2envmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_orthographic_gt --threads 25 

### to shceoff
python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_orthographic_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_orthographic_order100_gt --total_scene 1 --num_order 100

###### RERUN WITH PERSPECTIVE_DIV_4
python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_div4_gt --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_div4_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_div4_order100_gt --total_scene 1 --num_order 100

# order 6
python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_div4_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_div4v2_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_div4v2_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_div4v2_viz_ldr_gt --num_order 6

# order 2
python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_div4_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_viz_ldr_gt --num_order 2


# RERUN
python envmap2sh_hdr.py -t 2 -i 0



# RE-RUN COPYROOM10
python validate_2lora.py --dataset /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10 --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/0000000 --no_save_intermediate --no_torch_compile --no_random_loader --lora_path real_checkpoint/rev3/Flickr2K/Flickr2K_balanced_aligned/checkpoint-140000

python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/envmap_perspective_fov_gt --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/focal --threads 25 --fov_width 512

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_fov_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/{}/shcoeff_perspective_fov_order100_gt --total_scene 1 --num_order 100


python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_div4_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_div4v2_viz_ldr_gt --num_order 2


# RERUN AFTER IDENTIFY BUG
python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_v3 --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_v3 --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --total_scene 1 --num_order 100

python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_viz_ldr_gt --num_order 2

python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_viz_ldr_gt --num_order 6


python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3_gt_ball --num_order 100 --total_scene 1


python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_v3x2 --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512


#########################################################
# CREATE Bshading
##############################################

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/envmap_perspective_v3 --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/shcoeff_perspective_v3_order100 --total_scene 816 --num_order 100 --threads 8
