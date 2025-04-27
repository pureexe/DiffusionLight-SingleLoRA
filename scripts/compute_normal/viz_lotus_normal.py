import torch
import numpy as np 
from tqdm.auto import tqdm 
import os
import skimage 

SOURCE_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_lotus"
TARGET_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_lotus_viz_z_up_without_normalize"

def from_x_left_to_z_up(point):
    """
    Convert from ControlNet x-left, y-up, z-forward to x-forward, y-right z-up
    """
    assert point.shape[-1] == 3 # only support catesian coordinate
    rotation_matrix = np.array([
        [0., 0., 1.], # new x-forward  coming  from z-foward
        [-1., 0., 0.], # new y-right coming from x-left
        [0., 1., 0.], # new z-up comfing from y-up
    ])
    # convert to torch to multiply in last 2 dimension.
    rotation_matrix = torch.from_numpy(rotation_matrix).float()
    point =  torch.from_numpy(point)[...,None].float()
    new_point = rotation_matrix @ point 
    new_point = new_point[...,0].numpy() # shape [H,W,3]
    return new_point

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.chmod(TARGET_DIR, 0o777)
    files = sorted(os.listdir(SOURCE_DIR))
    for filename in tqdm(files):
        try:
            source_path = os.path.join(SOURCE_DIR, filename)
            target_path = os.path.join(TARGET_DIR, filename.replace('.npz','.png'))
            if os.path.exists(target_path):
                continue
            data = np.load(source_path)
            normal = data[data.files[0]]
            normal = from_x_left_to_z_up(normal)
            #normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = (normal + 1) / 2
            normal = np.clip(normal, 0, 1)
            normal = skimage.img_as_ubyte(normal)
            skimage.io.imsave(target_path, normal)
            os.chmod(target_path, 0o777)
        except Exception as e:
            continue
    
    
if __name__ == "__main__":
    main()