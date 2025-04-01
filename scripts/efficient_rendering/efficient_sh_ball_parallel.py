print("IMPORTING OS")
import os
from PIL import Image
from tqdm.auto import tqdm
import warnings
print("IMPORTING TORCH")
import torch
print("IMPORTING SKIMAGE")
import skimage
print("IMPORTING NUMPY")
import numpy as np
print("IMPORTING MULTIPROCESSING")
from multiprocessing import Pool, cpu_count
from functools import partial
import ezexr
print("IMPORT DONE")
import time
import argparse
from sh_utils import get_ideal_normal_ball_z_up, get_shcoeff, compute_background, sample_from_sh, unfold_sh_coeff, apply_integrate_conv, cartesian_to_spherical
from tonemapper import TonemapHDR

def process_scene(args, info):
        
    scene_name = info[0]
    filename = info[1]
                
    output_dir = args.output_dir_template.format(scene_name)
    os.makedirs(output_dir,exist_ok=True) 
    os.chmod(output_dir, 0o777)
        
    output_path = os.path.join(
        output_dir,
        filename + '.png'
    )
               
    if os.path.exists(output_path):
        return None
        
    # load normal map
    normal_map, mask = get_ideal_normal_ball_z_up(512) #THIS BALL IS SAME AS SPHERICAL HAMONIC RENDERING

    theta, phi = cartesian_to_spherical(normal_map)

    # load shcoeff 
    coeff_dir = args.coeff_dir_template.format(scene_name)
    coeff_path = os.path.join(
        coeff_dir, # (3, 10201)
        filename + '.npy'
    )
    shcoeff = np.load(coeff_path) # shcoeff shape (3,10201) (order-100)        
    shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=args.num_order) #(3,2,7,7) order 6

    shcoeff = apply_integrate_conv(shcoeff)
    shading = sample_from_sh(shcoeff, lmax=args.num_order, theta=theta, phi=phi)
    shading = shading / 10
    percentile_99 = np.percentile(shading,99)
    print(percentile_99)
    
    shading = np.clip(shading, 0, 1)
    shading = shading * mask[...,None]
    shading = skimage.img_as_ubyte(shading)
    skimage.io.imsave(output_path, shading)
    os.chmod(output_path, 0o777)
    
    return percentile_99

def efficient_rendering(args):
    queues = []
    for scene_id in tqdm(list(range(args.total_scene))[args.index::args.total]):
        scene_name = f"{scene_id*1000:06d}"
        coeff_dir = args.coeff_dir_template.format(scene_name)
        if not os.path.exists(coeff_dir):
            continue
        filenames = sorted([a for a in os.listdir(coeff_dir) if a.endswith('.npy')])
        for filename in filenames:
            queues.append((scene_name, filename.replace('.npy', ''), args))
    fn = partial(process_scene, args)
    with Pool(cpu_count()) as pool:
        a = list(tqdm(pool.imap_unordered(fn, queues), total=len(queues)))
        print(a)
        print("MAX PERCENTILE: ", np.array(a).max())
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process index and total.")
    parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
    parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')
    parser.add_argument('--coeff_dir_template', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shcoeff_perspective_fov_order100", help='template for coeff dir')
    parser.add_argument('--output_dir_template', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shading_exr_perspective_fov_order6", help='template for output dir')
    parser.add_argument('--total_scene', type=int, default=816)
    parser.add_argument('--num_order', type=int, default=6)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    efficient_rendering(args)