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
DATASET_PATH = "/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024"
SCENE_TEMPLATE = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/raw"

TOTAL_SCENE = 816

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    return parser

def main():
    args = create_argparser().parse_args()
    dir_ids = list(range(0, TOTAL_SCENE))
    dir_ids = dir_ids[args.idx::args.total]

    # load dataset
    pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    pipe_normal.set_progress_bar_config(disable=True)


    for dir_id in dir_ids:
        
        scene_id = f"{dir_id * 1000:06d}"
        chromeball_raw_dir =  SCENE_TEMPLATE.format(scene_id)
        if not os.path.exists(chromeball_raw_dir):
            continue
        output_dir = os.path.join(OUTPUT_DIR, scene_id)
        if not os.path.exists(output_dir):
            continue

        files_ids = sorted([a.replace('_ev-00.png','') for a in os.listdir(chromeball_raw_dir) if a.endswith('_ev-00.png')])
        normal_dir = os.path.join(output_dir, 'normal')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)
        os.chmod(normal_dir, 0o777)
        print("FOLDER_ID: ", scene_id)
        for file_id in tqdm(files_ids):
            try:
                npz_path = os.path.join(normal_dir, file_id+'.npz')
                if os.path.exists(npz_path):
                    continue
                image_path = os.path.join(DATASET_PATH, scene_id, file_id +'.jpg')
                image = skimage.io.imread(image_path)
                image = skimage.img_as_float(image)
                if len(image.shape) == 2:
                    image = np.concatenate([image[...,None],image[...,None],image[...,None]], axis=-1)
                image = torch.tensor(image).permute(2,0,1).to('cuda')
                normals = pipe_normal(image,output_type='pt').prediction
                normal_map = normals[0].cpu().permute(1,2,0).numpy()
                normal_map = normal_map.astype(np.float16)
                np.savez_compressed(npz_path, arr=normal_map)
                os.chmod(npz_path, 0o777)
            except Exception as e:
                print("ERROR: ", file_id)    
                print(f"An error occurred: ")
                print(e)
            

        
        
    
    
    
if __name__ == "__main__":
    main()