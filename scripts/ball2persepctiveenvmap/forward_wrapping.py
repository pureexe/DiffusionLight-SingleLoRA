# forward wrapping is for changing environment map to chromeball 

import numpy as np
from PIL import Image
import skimage
import time
import torch
import argparse 
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import os
import shutil
from tonemapper import TonemapHDR
from scipy.optimize import root
from scipy.optimize import least_squares

import math 

try:
    import ezexr
except:
    pass

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt" ,help='directory that contain the image') 
    parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal",help='directory that contain horizontal focal file.') 
    parser.add_argument("--envmap_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_reverse_unwrap" ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--envmap_height", type=int, default=256, help="size of the environment map height in pixel (height)")
    parser.add_argument("--ball_ratio", type=float, default=128 / 512, help="size of the environment map height in pixel (height)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--fov_width", type=int, default=512, help="size of image to calcurate focal")
    parser.add_argument("--threads", type=int, default=25, help="num thread for pararell processing")
    return parser


def main():
    # running time measuring
    start_time = time.time()        
    
    # make output directory if not exist
    os.makedirs(args.envmap_dir, exist_ok=True)
    os.chmod(args.envmap_dir, 0o777)
    
    # get all file in the directory
    files = sorted(os.listdir(args.ball_dir))

    # create partial function for pararell processing
    process_func = partial(process_image, args)
    process_func(files[2])
    exit()
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)
    
if __name__ == "__main__":
    main()