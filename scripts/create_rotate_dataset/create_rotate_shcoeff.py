import os 
import shutil
import numpy as np
from sh_utils import unfold_sh_coeff, flatten_sh_coeff, rotate_sh_coeff_x_axis
from tqdm.auto import tqdm
import argparse

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    shcoeff = np.load(args.source_shcoeff) #(3,10201)
    shcoeff = unfold_sh_coeff(shcoeff, max_sh_level=args.max_order)

    for step_id in tqdm(range(args.num_frame)):
        rotate_shcoeff = rotate_sh_coeff_x_axis(shcoeff, np.pi*2 * step_id / args.num_frame, lmax=args.max_order)
        flatten_rotate_shcoeff = flatten_sh_coeff(rotate_shcoeff, max_sh_level=args.max_order)
        print(flatten_rotate_shcoeff.shape)
        np.save(os.path.join(args.output_dir, f'dir_{step_id}_mip2.npy'), flatten_rotate_shcoeff)
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process index and total.")
    parser.add_argument('--source_shcoeff', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_dining1/shcoeff_perspective_v3_order100_main/dir_0_mip2.npy", help="source shcoeff")
    parser.add_argument('--output_dir', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_dining1/shcoeff_perspective_v3_order100", help="output shcoeff directory")
    parser.add_argument('--num_frame', type=int, default=60, help="number of frame that will create the video")
    parser.add_argument('--max_order', type=int, default=100, help="number of shcoeff_max_order")
    args = parser.parse_args()
    main(args)