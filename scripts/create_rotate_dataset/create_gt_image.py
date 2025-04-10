import os 
import shutil
import numpy as np
import skimage
from tqdm.auto import tqdm
import ezexr
import argparse

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    source_image = skimage.io.imread(args.source_image)
    source_image = skimage.img_as_float(source_image)
    source_shading = ezexr.imread(os.path.join(args.shadings_dir, 'dir_0_mip2.exr'))
    source_shading = np.clip(source_shading, 0, np.inf)
    albedo = source_image / (source_shading + 1e-8)
    
    for step_id in tqdm(range(args.num_frame)):
        shading = ezexr.imread(os.path.join(args.shadings_dir, f'dir_{step_id}_mip2.exr'))
        image = albedo * shading
        image = np.clip(image, 0, 1)
        image = skimage.img_as_ubyte(image)
        skimage.io.imsave(os.path.join(args.output_dir, f'dir_{step_id}_mip2.png'),image)

    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process index and total.")
    parser.add_argument('--source_image', type=str, default="/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10/dir_18_mip2.jpg", help="source shcoeff")
    parser.add_argument('--shadings_dir', type=str, default="/ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/shadings/14n_copyroom10_light18", help="output shcoeff directory")
    parser.add_argument('--output_dir', type=str, default="/ist/ist-share/vision/relight/datasets/laion-shading/v2/rotate/images/14n_copyroom10_light18", help="output shcoeff directory")
    parser.add_argument('--num_frame', type=int, default=60, help="number of frame that will create the video")
    args = parser.parse_args()
    main(args)