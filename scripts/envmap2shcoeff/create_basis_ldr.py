import os 

def main():
    sh_coeffs = [2, 3, 4, 6, 10, 20, 50, 100]
    for sh_order in sh_coeffs:
        cmd = f'python envmap2sh_hdr.py --input_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/envmap_perspective_v3_ldr_gt --output_template /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/{{}}/basis_shcoeff_ldr_gt/order{sh_order} --total_scene 1 --num_order {sh_order}'
        os.system(cmd)

if __name__ == "__main__":
    main()

