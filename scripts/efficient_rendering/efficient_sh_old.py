print("IMPORTING OS")
import os
from PIL import Image
print("IMPORTING PIPELINE")
from transformers import pipeline
from tqdm.auto import tqdm
import warnings
print("IMPORTING TORCH")
import torch
print("IMPORTING SKIMAGE")
import skimage
print("IMPORTING NUMPY")
import numpy as np
print("IMPORTING CV2")
import cv2
print("IMPORTING THE NormalBAE")
from controlnet_aux import NormalBaeDetector
print("IMPORTING PYSHTOOLS")
import pyshtools
print("IMPORTING MULTIPROCESSING")
from multiprocessing import Pool
from functools import partial
print("IMPORTING CONTROLNET_UTIL")
from controlnet_aux.util import HWC3, resize_image
print("IMPOORING EINOPS")
from einops import rearrange
from tonemapper import TonemapHDR
import ezexr
print("IMPORT DONE")
import time
import cv2
import argparse
from shading_integrate import get_envmap_from_file
from sh_utils import get_ideal_normal_ball_z_up, cartesian_to_spherical, get_shcoeff, compute_background, sample_from_sh, from_x_left_to_z_up, unfold_sh_coeff, apply_integrate_conv

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')

args = parser.parse_args()

OUTPUT_DIR = "output/efficient_rendering_10k"
ORDER = 2


class NormalBaeDetectorPT(NormalBaeDetector):
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            #normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            #normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return normal


def efficient_rendering_chromeball():
    image_width = 512
    theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
    phi = np.linspace(0, np.pi * 2, 2*image_width)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')

    USE_INTEGRATE_CONV = True
    SAVE_NORMAL_MAP = True
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image = get_envmap_from_file()
    shcoeff = get_shcoeff(image, Lmax=ORDER)
    assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2 and shcoeff.shape[2] ==  ORDER+1 and shcoeff.shape[3] ==  ORDER+1 # make sure shape [3,2,3,3]
    
    if USE_INTEGRATE_CONV:
        integrated_shcoeff = apply_integrate_conv(shcoeff)
    else:
        integrated_shcoeff = shcoeff
    normal_map, mask = get_ideal_normal_ball_z_up(256) #THIS BALL IS SAME AS SPHERICAL HAMONIC RENDERING
    if SAVE_NORMAL_MAP:
        normal_ball_image = (normal_map + 1 / 2.0)
        normal_ball_image = skimage.img_as_ubyte(np.clip(normal_ball_image, 0, 1))
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "normal_ball.png"), normal_ball_image)

    theta, phi = cartesian_to_spherical(normal_map) 

    shading = sample_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
    # tonemapped_shading = np.clip(shading, 0, 1)
    ezexr.imwrite(os.path.join(OUTPUT_DIR, "rendered_shading.exr"), shading)
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    tonemapped_shading, _, _ = tonemap(shading)
    skimage.io.imsave(os.path.join(OUTPUT_DIR, "tonemapped_shading.png"), skimage.img_as_ubyte(tonemapped_shading))
    print(f"Time elapsed: {time.time() - start_time}")

def efficeint_rendering():
    print("LOADING PREPROCESSOR")
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test"
    image_dir = "images"
    coeff_dir = "shcoeffs_order100_hdr_v2"
    output_dir = "control_shading_from_hdr27coeff_conv_v5"
    mode = 'bae'
    ORDER = 2

    os.makedirs(os.path.join(root_dir, output_dir), exist_ok=True)
    os.chmod(os.path.join(root_dir, output_dir), 0o777)

    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor.to('cuda')

    print("CREATING QUEUES...")
    queues  = []
    for scene in scenes:
        for idx in range(25):
            queues.append((scene,idx))
    
    tonemapper = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    pbar = tqdm(queues[args.index::args.total])
    pbar.set_description(f"")

    for info in pbar:
        
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")

        idx = info[1]
        scene = info[0]
        shading_output_dir = os.path.join(root_dir,output_dir,scene)
        output_path = os.path.join(shading_output_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(output_path):
           continue
        image = Image.open(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        normal_map = preprocessor(image, output_type="pt")
        normal_map = from_x_left_to_z_up(normal_map)

        theta, phi = cartesian_to_spherical(normal_map)


        shcoeff = np.load(f"{root_dir}/{coeff_dir}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        
        shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=ORDER)

        shcoeff = apply_integrate_conv(shcoeff)

        shading = sample_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
        
        os.makedirs(shading_output_dir, exist_ok=True)
        os.chmod(shading_output_dir, 0o777)

        #ezexr.imwrite(output_path.replace(".png",".exr"), shading)   
        shading = np.float32(shading)
        shading, _, _ = tonemapper(shading) # tonemap
        #shading = tonemapper.process(shading)

        
        shading = np.clip(shading, 0, 1)
        shading = skimage.img_as_ubyte(shading)
        try:
            skimage.io.imsave(output_path,shading)
            os.chmod(output_path, 0o777)
        except:
            pass



def inspect_football_normal():
    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor.to('cuda')
    image = Image.open(f"data/ball1.png").convert("RGB")
    normal_map = preprocessor(image, output_type="pt")
    np.save("output/efficient_rendering/ball1.npy", normal_map)




if __name__ == "__main__":
    #efficeint_rendering()
    efficient_rendering_chromeball()