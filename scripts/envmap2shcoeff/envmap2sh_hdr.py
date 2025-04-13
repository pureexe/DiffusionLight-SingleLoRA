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


#TOTAL_SCENE = 816
#INPUT_DIR = 
#OUTPUT_DIR = 
#ORDER = 100

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total", type=int, default=1, help="total process")
    parser.add_argument("-i","--idx", type=int, default=0, help="process id")
    parser.add_argument("--input_template", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/envmap_perspective_fov", help="template path for input dir")
    parser.add_argument("--output_template", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shcoeff_perspective_fov_order100", help="template path for output dir")
    parser.add_argument("--total_scene", type=int, default=816, help="number of total scene")
    parser.add_argument("--num_order", type=int, default=100, help="number of sperical harmonic order")
    parser.add_argument("--threads", type=int, default=40, help="number of threads")
    return parser

def process_file(meta):
    input_dir, output_dir, filename, num_order = meta
    _, file_extension = os.path.splitext(filename)
    out_path = os.path.join(output_dir, filename.replace(file_extension, ".npy"))
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
    args = create_argparser().parse_args()
    scene_ids = list(range(0, args.total_scene))
    scene_ids = scene_ids[args.idx::args.total]
    for scene_id in scene_ids:
        print("SCENE: ", scene_id)
        # running time measuring
        start_time = time.time()        
        
        scene_name = f"{scene_id * 1000:06d}"
        input_dir = args.input_template.format(scene_name)
        output_dir = args.output_template.format(scene_name)
        
        if not os.path.exists(input_dir):
            print(input_dir)
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)
        
        # get all scenes 
        scenes = sorted([a for a in os.listdir(input_dir)]) #if a.endswith('.exr')

        queues = []
        # generate all queue
        print("QUEUEING SCENE...")
        for scene in tqdm(scenes):
            queues.append([input_dir, output_dir, scene, args.num_order])

        print("PROCESSING FILE...")
        with Pool(args.threads) as p:
            list(tqdm(p.imap(process_file, queues), total=len(queues)))
        
        # print total time 
        print("TOTAL TIME: ", time.time() - start_time)





if __name__ == "__main__":
    main()