import os 
def main():
    sh_coeffs = [2, 3, 4, 6, 10, 20, 50, 100]
    for sh_order in sh_coeffs:
        print("SH ORDER: ", sh_order)
        os.system(f'python efficient_sh.py --coeff_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/basis_shcoeff_ldr_gt/order{sh_order} --normal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/normal --focal_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/focal --output_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/c_result/shading_exr_order{sh_order}diffuse_v3a_ball_ldr_gt  --vizmax_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/c_result/order{sh_order}_viz_max_gt --vizldr_dir_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/c_result/order{sh_order}_viz_ldr_gt --num_order {sh_order} --image_width 512 --image_height 512 --fov_width 512 --total_scene 1 --use_ball 1 --use_lotus 0 --apply_integrate 1 --use_viz 1 --threads 25')
if __name__ == "__main__":
    main()