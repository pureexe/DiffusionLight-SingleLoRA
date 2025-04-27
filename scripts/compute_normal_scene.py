import torch
import diffusers
import numpy as np
import os 
from PIL import Image 
import torchvision
import argparse 
import skimage 
from tqdm.auto import tqdm 
import numpy as np
import ezexr 

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyshtools

MASTER_TYPE = torch.float16
# DATASET_PATH = "/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images"
# OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024"
# SCENE_TEMPLATE = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/raw"
# TOTAL_SCENE = 816

IMAGE_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10"
NORMAL_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/normal"
VIZ_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom10/000000/normal_viz"

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    return parser

def main():
    args = create_argparser().parse_args()

    # load dataset
    pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    pipe_normal.set_progress_bar_config(disable=True)

    files = sorted([a for a in os.listdir(IMAGE_PATH) if a.endswith('.jpg')])
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.chmod(NORMAL_DIR, 0o777)
    os.chmod(VIZ_DIR, 0o777)

    for filename in files:
        
        nomral_path = os.path.join(NORMAL_DIR, filename.replace('.jpg','.npz'))
        viz_path = os.path.join(VIZ_DIR, filename.replace('.jpg','.png'))

        # if os.path.exists(nomral_path):
        #     continue
        image_path = os.path.join(IMAGE_PATH, filename)
        image = skimage.io.imread(image_path)
        image = skimage.img_as_float(image)
        if len(image.shape) == 2:
            image = np.concatenate([image[...,None],image[...,None],image[...,None]], axis=-1)
        image = torch.tensor(image).permute(2,0,1).to('cuda')
        normals = pipe_normal(image,output_type='pt').prediction
        normal_map = normals[0].cpu().permute(1,2,0).numpy()
        normal_map = normal_map.astype(np.float16)
        np.savez_compressed(nomral_path, arr=normal_map)
        os.chmod(nomral_path, 0o777)
        # viz 
        normal_map = (normal_map + 1) / 2
        normal_map = np.clip(normal_map, 0, 1)
        normal_map = skimage.img_as_ubyte(normal_map)
        skimage.io.imsave(viz_path, normal_map)
        os.chmod(viz_path, 0o777)
            

        
        
    
    
    
if __name__ == "__main__":
    main()