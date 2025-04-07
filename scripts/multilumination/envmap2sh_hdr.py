# create a spherical harmonic from a chromeball image

import os
import numpy as np 
import skimage
import time
import ezexr
from multiprocessing import Pool 
from tqdm.auto import tqdm
from sh_utils import get_shcoeff, flatten_sh_coeff

import argparse


SPLIT_TYPE = 'train'


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total", type=int, default=1, help="total process")
    parser.add_argument("-i","--idx", type=int, default=0, help="process id")
    parser.add_argument("--input_template", type=str, default=f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT_TYPE}/envmap_perspective", help="template path for input dir")
    parser.add_argument("--output_template", type=str, default=f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT_TYPE}/shcoeff_perspective_order100", help="template path for output dir")
    parser.add_argument("--total_scene", type=int, default=816, help="number of total scene")
    parser.add_argument("--num_order", type=int, default=100, help="number of sperical harmonic order")
    parser.add_argument("--threads", type=int, default=40, help="number of threads")
    return parser

def process_file(meta):
    input_dir, output_dir, filename, num_order = meta
    out_path = os.path.join(output_dir, filename.replace(".exr", ".npy"))
    if os.path.exists(out_path):
        return None
    in_path = os.path.join(input_dir, filename)
    if not os.path.exists(in_path):
        return None
    try:
        image = ezexr.imread(in_path)
    except:
        return None
    image = skimage.img_as_float(image)[...,:3]
    image = np.clip(image,0,np.inf)
    coeff = get_shcoeff(image, Lmax=num_order)
    shcoeff = flatten_sh_coeff(coeff, max_sh_level=num_order)
    np.save(out_path, shcoeff)
    os.chmod(output_dir, 0o777)
    return None


def main():
    start_time = time.time()
    args = create_argparser().parse_args()
    scene_ids = os.listdir(args.input_template)

    queues = []
    for scene_name in scene_ids:
        input_dir = os.path.join(args.input_template, scene_name)
        output_dir = os.path.join(args.output_template, scene_name)
        files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.exr')]
        for filename in files:
            queues.append([input_dir, output_dir, filename, args.num_order])
        
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)
    
    print("PROCESSING FILE...")
    with Pool(40) as p:
        list(tqdm(p.imap(process_file, queues), total=len(queues)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)





if __name__ == "__main__":
    main()