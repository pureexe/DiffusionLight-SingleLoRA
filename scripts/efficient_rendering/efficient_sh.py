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
from multiprocessing import Pool
from functools import partial
import ezexr
print("IMPORT DONE")
import time
import argparse
from sh_utils import get_ideal_normal_ball_z_up, cartesian_to_spherical, get_shcoeff, compute_background, sample_from_sh, from_y_up_to_z_up, unfold_sh_coeff, apply_integrate_conv
from tonemapper import 

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')

args = parser.parse_args()

COEFF_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shcoeff_perspective_order100"
NORMAL_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/normal"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shading_exr_order6"
VIZMAX_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shading_exr_order6_viz_max"
VIZLDR_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shading_exr_order6_viz_ldr"
USE_VIZ = True
ORDER = 6
TOTAL_SCENE = 816

def efficeint_rendering():
    scene_id = 0
    queues = []     
    print('bulding index...')
    for scene_id in tqdm(list(range(TOTAL_SCENE))[args.index::args.total]):
        scene_name = f"{scene_id*1000:06d}"    
        coeff_dir = COEFF_DIR.format(scene_name)
        if not os.path.exists(coeff_dir):
            continue
        filenames = sorted([a for a in os.listdir(coeff_dir) if a.endswith('.npy')])
        for filename in filenames:               
            queues.append([
                scene_name,
                filename.replace('.npy','')
            ])
            
    pbar = tqdm(queues)
    pbar.set_description(f"")
    
    # TODO: NEED TO MAKE THIS FOR LOOP PARARELL
    for info in pbar:
        
        scene_name = info[0]
        filename = info[1]
        
        pbar.set_postfix(item=f"{filename}")
        
        output_dir = OUTPUT_DIR.format(scene_name)
        os.makedirs(output_dir,exist_ok=True) 
        os.chmod(output_dir, 0o777)
        
        output_path = os.path.join(
            output_dir,
            filename + '.exr'
        )
               
        if os.path.exists(output_path):
           continue

        normal_dir = NORMAL_DIR.format(scene_name)        
        normal_path = os.path.join(
            normal_dir,
            filename + '.npz'
        )
        
        # load normal map
        try:
            normal_map = np.load(normal_path)
            normal_map = normal_map[normal_map.files[0]]
        except:
            continue
        normal_map = from_y_up_to_z_up(normal_map)
        theta, phi = cartesian_to_spherical(normal_map)

        # load shcoeff 
        coeff_dir = COEFF_DIR.format(scene_name)
        coeff_path = os.path.join(
            coeff_dir, # (3, 10201)
            filename + '.npy'
        )
        shcoeff = np.load(coeff_path) # shcoeff shape (3,10201) (order-100)        
        shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=ORDER) #(3,2,7,7) order 6
  
        shcoeff = apply_integrate_conv(shcoeff)
        shading = sample_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
        
        shading = np.float32(shading)
        ezexr.imwrite(output_path, shading)
        os.chmod(output_path, 0o777)
        
        if USE_VIZ:
            vizmax_dir = VIZMAX_DIR.format(scene_name)
            vizldr_dir = VIZLDR_DIR.format(scene_name)
            
            os.makedirs(vizmax_dir)
            os.chmod(vizmax_dir, 0o777)
            os.makedirs(vizldr_dir)
            os.chmod(vizldr_dir, 0o777)
            
            tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
            tonemapped_shading, _, _ = tonemap(shading)
            vizldr_path = os.path.join(vizldr_dir, filename + '.png')
            skimage.io.imsave(vizldr_path, skimage.img_as_ubyte(tonemapped_shading))
            os.chmod(vizldr_path, 0o777)
            
            percentile99 = np.percentile(shading, 99)
            shading_norm = shading / percentile99
            shading_norm = np.clip(shading_norm, 0, 1)
            vizmax_path = os.path.join(vizmax_dir, filename + '.png')
            skimage.io.imsave(vizmax_path, skimage.img_as_ubyte(shading_norm))
            os.chmod(vizmax_path, 0o777)
            
        
if __name__ == "__main__":
    efficeint_rendering()
