# create a copy of normal map from source normal map for efficient rendering code

import os 
import shutil
import numpy as np
from tqdm.auto import tqdm
import argparse

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    for step_id in tqdm(range(args.num_frame)):
        shutil.copy2(args.source_normal, os.path.join(args.output_dir, f'dir_{step_id}_mip2.npz'))
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process index and total.")
    parser.add_argument('--source_normal', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/normal_lotus/dir_18_mip2.npz", help="source shcoeff")
    parser.add_argument('--output_dir', type=str, default="/ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/normal_lotus/14n_copyroom10_light18", help="output shcoeff directory")
    parser.add_argument('--num_frame', type=int, default=60, help="number of frame that will create the video")
    args = parser.parse_args()
    main(args)