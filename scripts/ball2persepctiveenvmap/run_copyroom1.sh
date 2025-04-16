
# UNWRAP
python reverse_wrapping.py --focal_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal  --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_v3_gt --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt --fov_width 512 --threads 1

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_v3_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --total_scene 1 --num_order 100

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_v3_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order6_gt --total_scene 1 --num_order 6


python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3_viz_ldr_gt --num_order 6


python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_viz_ldr_gt --num_order 2


###############
python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3_ball_gt  --num_order 2


python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3_ball_gt  --num_order 100

##### RE PREDICT
python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_viz_ldr_gt --num_order 100 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --apply_integrate 0 --threads 25


python efficient_sh_lotus.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal_lotus --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_viz_ldr_gt --num_order 6 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 0 --apply_integrate 0 --threads 25

# look into normal
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6_v3a_viz_ldr_gt --num_order 6 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 0 --use_lotus 0 --apply_integrate 0 --threads 25 

# create order 2
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_viz_ldr_gt --num_order 2 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 0 --use_lotus 0 --apply_integrate 0 --threads 25 


# create ball
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100_v3a_ball_viz_ldr_gt --num_order 100 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --use_lotus 0 --apply_integrate 0 --threads 25 


# create ball_order 2
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order2_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_ball_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_ball_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order2_v3a_ball_viz_ldr_gt --num_order 2 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --use_lotus 0 --apply_integrate 0 --threads 25 

# create ball_order 6
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6diffuse_v3a_ball_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order6diffuse_v3a_ball_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order100diffuse_v3a_ball_viz_ldr_gt --num_order 6 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --use_lotus 0 --apply_integrate 1 --threads 25 

# order 20
python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_gt --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order3diffuse_v3a_ball_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order3diffuse_v3a_ball_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shading_exr_order3diffuse_v3a_ball_viz_ldr_gt --num_order 3 --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --use_lotus 0 --apply_integrate 1 --threads 25 



########### LDR test
python reverse_wrapping.py --focal_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal  --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective_v3_ldr_gt --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_ldr_gt --fov_width 512 --threads 1

python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_v3_ldr_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_ldr_gt --total_scene 1 --num_order 100

#/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/shcoeff_perspective_v3_order100_ldr_gt


# RUN LDR BASIS
python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective_v3_ldr_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_v3_order100_ldr_gt --total_scene 1 --num_order 100
