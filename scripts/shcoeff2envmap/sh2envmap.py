# create a spherical harmonic from a chromeball image

import os
import numpy as np 
import skimage
import time
import ezexr
from multiprocessing import Pool 
from functools import partial
from tqdm.auto import tqdm
from sh_utils import get_shcoeff, unfold_sh_coeff,compute_background

import argparse


#TOTAL_SCENE = 816
#INPUT_DIR = 
#OUTPUT_DIR = 
#ORDER = 100

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total", type=int, default=1, help="total process")
    parser.add_argument("-i","--idx", type=int, default=0, help="process id")
    parser.add_argument("--input_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate", help="template path for input dir")
    parser.add_argument("--output_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate", help="template path for output dir")
    parser.add_argument("--envmap_dir", type=str, default="envmap_perspective_v3_order2", help="envmap dir")
    parser.add_argument("--shcoeff_dir", type=str, default="shcoeff_perspective_v3_order2", help="shcoeff dir")
    parser.add_argument("--num_order", type=int, default=2, help="number of sperical harmonic order")
    parser.add_argument("--threads", type=int, default=8, help="number of threads")
    parser.add_argument("--value_scaleup", type=float, default=1.0, help="value scale up")
    return parser

def process_file(args, meta):

    scene, filename = meta
    input_dir = os.path.join(args.input_dir, scene, args.shcoeff_dir)
    out_dir = os.path.join(args.input_dir, scene, args.envmap_dir)
    os.makedirs(out_dir,exist_ok=True)
    os.chmod(out_dir, 0o777)

    _, file_extension = os.path.splitext(filename)
    out_path = os.path.join(out_dir, filename.replace(file_extension, ".exr"))
    if os.path.exists(out_path):
        return None
    in_path = os.path.join(input_dir, filename)
    if not os.path.exists(in_path):
        return None
    try:
        shcoeff = np.load(in_path)
    except:
        return None
    shcoeff = unfold_sh_coeff(shcoeff, max_sh_level=args.num_order)
    envmmap = compute_background(hfov=None,sh=shcoeff,lmax=args.num_order)
    envmmap = envmmap[...,:3] * args.value_scaleup
    print(out_path)
    print("MAX: VALUE ")
    print(envmmap.max())
    ezexr.imwrite(out_path, envmmap)
    os.chmod(out_dir, 0o777)
    return None


def main():
    args = create_argparser().parse_args()
    # seek file
    
    queues = []

    print("seeking file...")
    scenes = os.listdir(args.input_dir)
    for scene_name in tqdm(scenes):
        input_dir = os.path.join(args.input_dir, scene_name, args.shcoeff_dir)
        avalible_files = sorted(os.listdir(input_dir))
        for fname in avalible_files:
            queues.append(
                [scene_name, fname]
            )

    queues = queues[args.idx::args.total]
    fn = partial(process_file, args)
    print("PROCESSING FILE...")
    with Pool(args.threads) as p:
        list(tqdm(p.imap(fn, queues), total=len(queues)))
    

        # os.makedirs(output_dir, exist_ok=True)
        # os.chmod(output_dir, 0o777)
        




if __name__ == "__main__":
    main()