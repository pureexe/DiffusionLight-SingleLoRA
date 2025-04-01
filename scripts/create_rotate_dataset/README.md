# CREATE ROTATE DATASET INSTRUCTION
1. create new sh using create_rotate_shcoeff.py
2. select source normal and copy for all normal map using copy_normal.py

3. create new shading using efficient_rendering

```
python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/14n_copyroom10_light18 --normal_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/14n_copyroom10_light18 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/14n_copyroom10_light18 --vizmax_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_max/14n_copyroom10_light18 --vizldr_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_ldr/14n_copyroom10_light18 --total_scene 1
```

4. create the gt image

5. create normal ball 
```
python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/14n_copyroom10_light18 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/ball/14n_copyroom10_light18 --total_scene 1
```

## copyroom1 scene
1. create_rotate_shcoeff.py 

```
python create_rotate_shcoeff.py --source_shcoeff /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/shcoeff_perspective_div4_order100_gt/dir_24_mip2.npy --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/14n_copyroom1_light24
```

2. select source normal and copy for all normal map using copy_normal.py

```
python copy_normal.py --source_normal /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/normal_lotus/dir_24_mip2.npz --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/14n_copyroom1_light24
```

3. create new shading using efficient_rendering

```
python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/14n_copyroom1_light24 --normal_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/14n_copyroom1_light24 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/14n_copyroom1_light24 --vizmax_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_max/14n_copyroom1_light24 --vizldr_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_ldr/14n_copyroom1_light24 --total_scene 1
```

4. create the gt image
```
python create_gt_image.py --source_image /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom1/dir_24_mip2.jpg --shadings_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/14n_copyroom1_light24 --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/images/14n_copyroom1_light24
```

5. create normal ball 
```
python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/14n_copyroom1_light24 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/ball/14n_copyroom1_light24 --total_scene 1
```


## 000037
1. create_rotate_shcoeff.py 

```
python create_rotate_shcoeff.py --source_shcoeff /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/shcoeff_perspective_fov_order100/000037.npy --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000037
```
2. select source normal and copy for all normal map using copy_normal.py

```
python copy_normal.py --source_normal /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_lotus/000037.npz --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/000037
```
3. create new shading using efficient_rendering

```
python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000037 --normal_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/000037 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/000037 --vizmax_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_max/000037 --vizldr_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_ldr/000037 --total_scene 1
```

4. create the gt image
```
python create_gt_image.py --source_image /ist/ist-share/vision/relight/datasets/laion-shading/v2/train/images/000000/000037.jpg --shadings_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/000037 --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/images/000037
```

5. create normal ball 
```
python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000037 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/ball/000037 --total_scene 1
```

## 000071

1. create_rotate_shcoeff.py 

```
python create_rotate_shcoeff.py --source_shcoeff /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/shcoeff_perspective_fov_order100/000071.npy --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000071
```
2. select source normal and copy for all normal map using copy_normal.py

```
python copy_normal.py --source_normal /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_lotus/000071.npz --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/000071
```
3. create new shading using efficient_rendering

```
python efficient_sh_lotus_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000071 --normal_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/000071 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/000071 --vizmax_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_max/000071 --vizldr_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings_ldr/000071 --total_scene 1
```

4. create the gt image
```
python create_gt_image.py --source_image /ist/ist-share/vision/relight/datasets/laion-shading/v2/train/images/000000/000071.jpg --shadings_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/000071 --output_dir /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/images/000071
```

5. create normal ball 
```
python efficient_sh_ball_parallel.py --coeff_dir_template /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shcoeffs/000071 --output_dir_template  /ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/ball/000071 --total_scene 1
```
