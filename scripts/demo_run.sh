# load env
bin/diffusionlight_shell

# create chromeball
python validate_2lora.py --dataset /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom1 --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000 --no_save_intermediate --no_torch_compile --no_random_loader --lora_path real_checkpoint/rev3/Flickr2K/Flickr2K_balanced_aligned/checkpoint-140000

# create HDR chromeball
python exposure2hdr.py --input_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000/square --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/0000000/square_hdr

# predict normal
python copy1_compute_normal.py

# predict depth and fov
python /ist/ist-share/vision/pakkapon/relight/ml-depth-pro/run_copyroom1.py

# create perspective projection
python ball2perspectiveenvmap.py --ball_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr --envmap_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_perspective --fov_dir  /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal --threads 25 --fov_width 512

# create SHCOEFF
python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/envmap_perspective --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{}/shcoeff_perspective_order100 --total_scene 1 --num_order 100

# create shading