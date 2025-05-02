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


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    parser.add_argument("--input_dir", type=str, default='/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/rotate/images')
    parser.add_argument("--output_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate")
    parser.add_argument("--dir_name", type=str, default="normal")
    return parser

def main():
    args = create_argparser().parse_args()
    dir_ids = sorted(os.listdir(args.input_dir))
    #dir_ids = ['everett_dining1']
    dir_ids = dir_ids[args.idx::args.total]

    # load dataset
    pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    pipe_normal.set_progress_bar_config(disable=True)


    for scene_id in dir_ids:
        
        input_dir = os.path.join(args.input_dir, scene_id)
        output_dir = os.path.join(args.output_dir, scene_id)
        # if not os.path.exists(output_dir):
        #     continue

        files_ids = sorted([a.replace('.jpg','') for a in os.listdir(input_dir) if a.endswith('.jpg')])
        #files_ids = sorted([a.replace('.npy','') for a in os.listdir(chromeball_raw_dir) if a.endswith('.npy')])
        normal_dir = os.path.join(output_dir, args.dir_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)
        os.chmod(normal_dir, 0o777)
        for file_id in tqdm(files_ids):
            # try:
            if True:
                npz_path = os.path.join(normal_dir, file_id+'.npz')
                if os.path.exists(npz_path):
                    continue
                image_path = os.path.join(args.input_dir, scene_id, file_id +'.jpg')
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
            # except Exception as e:
            #     print("ERROR: ", file_id)    
            #     print(f"An error occurred: ")
            #     print(e)
            

        
        
    
    
    
if __name__ == "__main__":
    main()