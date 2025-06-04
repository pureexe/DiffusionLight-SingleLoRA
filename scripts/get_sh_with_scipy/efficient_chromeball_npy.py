import os
from PIL import Image
from tqdm.auto import tqdm
import warnings
import torch
import skimage
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import ezexr
import time
import argparse
from sh_utils import get_ideal_normal_ball_z_up, get_shcoeff, compute_background, sample_from_sh, unfold_sh_coeff, apply_integrate_conv, from_x_left_to_z_up, from_y_up_to_z_up, cartesian_to_spherical
from tonemapper import TonemapHDR

def get_fov(args, scene_name, filename):
    focal_dir = args.focal_dir_template.format(scene_name)
    new_filename = filename.split('.')[0]+'.npy'
    focal_path = os.path.join(focal_dir, new_filename)
    focal_px = np.load(focal_path) # focal length in term of pxiel
    fov_rad = 2 * np.arctan2(args.fov_width, 2*focal_px)
    return fov_rad

def get_viewing_directions(H, W, fov_rad_x):
    """
    @params
    - H: height of the image
    - W: width of the image
    - fov_rad_x: horizontal field of view in degrees (ml-depth-pro is predict in horizontal direction)
    @return
    - viewing_direction: viewing direction in cartesian coordinates (x-forward, y-right, z-up)
    """
    # focal length in pixel
    aspect_ratio = W / H
    fy = W / (2 * np.tan(fov_rad_x / 2))
    fz = fy / aspect_ratio    
    
    # create pixel grid in NDC space [-1,1]
    y = torch.arange(0, W) # [0,size-1] # this value will act as a middle pixel is on the edge, but in the environment map we need corner to be the edge
    y = (y + 0.5) / W # [0.5 / size, size - 0.5 / size]
    y = y * 2 - 1 # rescale to [-1,1]

    z = torch.arange(0, H) # [0,size-1] # this value will act as a middle pixel is on the edge, but in the environment map we need corner to be the edge
    z = z / H # rescale tp [0,1]
    z = z * 2 - 1 # rescale to [-1,1]
    z = torch.flip(z, dims=[0]) # flip the z axis to match the image coordinate system
    
    # convert to camera space 
    
    yy, zz = torch.meshgrid(y, z ,indexing='xy')     
    
    viewing_directions = torch.stack(
        (
            -torch.ones_like(yy),
            yy / fy,
            zz / fz
        ), 
        dim=-1
    ) # (H,W,3)
    # normalize to unit length
    viewing_directions = viewing_directions / torch.linalg.norm(viewing_directions, dim=-1, keepdim=True) # normalize to unit length
    viewing_directions = viewing_directions.cpu().numpy() # convert to numpy array
    return viewing_directions

def get_reflect_directions(viewing_directions, normal_directions):
    """
    Reflect viewing_directions around normal_directions.
    Supports broadcasting for any shape ending in 3.
    Formula is R = V - 2 * (V . N) * N
    
    Args:
        viewing_directions: np.ndarray of shape (..., 3)
        normal_directions:   np.ndarray of shape (..., 3)
        
    Returns:
        np.ndarray of shape (..., 3) representing the reflection direction.
    """
    # make sure it in cartesian coordinates
    assert viewing_directions.shape[-1] == 3
    assert normal_directions.shape[-1] == 3

    
    # Compute dot product and reflection
    dot = np.sum(viewing_directions * normal_directions, axis=-1, keepdims=True)
    reflect_directions = viewing_directions - 2 * dot * normal_directions
    return reflect_directions

def ignore_z_axis(N):
    x,y,oz = N[...,0], N[...,1], N[...,2]
    s = np.sign(oz)
    z = np.sqrt(1 - x**2 - y**2)
    z = s * z
    # concatenate the z axis to the x and y axis
    return np.concatenate((x[...,None], y[...,None], z[...,None]), axis=-1)
    

def process_scene(args, info):

    scene_name = info[0]
    filename = info[1]
                
    #output_dir = args.output_dir_template.format(scene_name)
    output_dir = os.path.join(args.output_dir, scene_name, args.shading_dir)
    try:
        os.makedirs(output_dir,exist_ok=True) 
        os.chmod(output_dir, 0o777)
    except:
        pass
        
    output_path = os.path.join(
        output_dir,
        filename + '.exr'
    )
               
    if os.path.exists(output_path):
        return None

    normal_filename = 'dir_0_mip2' if args.use_rotate_normal == 1 else filename
    normal_dir = os.path.join(args.output_dir, scene_name, args.normal_dir)      
    normal_path = os.path.join(
        normal_dir,
        normal_filename + '.npz'
    )
        
    # load normal map
    if args.use_ball == 1:
        # load normal map
        normal_directions, mask = get_ideal_normal_ball_z_up(512) #THIS BALL IS SAME AS SPHERICAL HAMONIC RENDERING
    else:
        try:
            normal_map = np.load(normal_path)
            normal_map = normal_map[normal_map.files[0]]
        except:
            print("NO NORMAL")
            return None
        # normmalize normal map
        normal_map = normal_map.astype(np.float32)
        if args.use_lotus == 1:
            normal_map = normal_map / np.linalg.norm(normal_map, axis=-1, keepdims=True)  # re normalize to unit length
            normal_directions = from_x_left_to_z_up(normal_map) # convert from Lotus convention (x-left.y-up,z-forward) to pyshtool (x-right,y-forward,z-up) 
        else:
            # Marigold use x-right, y-up, z-forward
            # @see https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage        
            normal_map = normal_map / np.linalg.norm(normal_map, axis=-1, keepdims=True)  # re normalize to unit length   
            normal_directions = from_y_up_to_z_up(normal_map)
            #normal_directions[..., 2] *= -1 # flip the y axis to match the image coordinate system
        mask = None
    
    if args.use_single_ray_trace == 1:
        # we need to convert normal to reflection vector first 
        fov = get_fov(args, scene_name, normal_filename)
        viewing_directions = get_viewing_directions(args.image_height, args.image_width, fov)
        reflect_directions = get_reflect_directions(viewing_directions, normal_directions)
        theta, phi = cartesian_to_spherical(reflect_directions)
    else:
        # here we assume that surface is diffuse.
        # Diffuse is combine all light direction around normal vector (diffuse lobe) regardless of viewing or reflect light direction 
        # @see https://imgur.com/a/Xtga9Mq
        theta, phi = cartesian_to_spherical(normal_directions)
       
    # load shcoeff 
    coeff_dir = os.path.join(args.output_dir, scene_name, args.coeff_dir)
    coeff_path = os.path.join(
        coeff_dir, # (3, 10201)
        filename + '.npy'
    )
    shcoeff = np.load(coeff_path) # shcoeff shape (3,10201) (order-100)        
    shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=args.num_order) #(3,2,7,7) order 6

    if args.apply_integrate == 1:
        shcoeff = apply_integrate_conv(shcoeff, lmax=args.num_order)
    
    shading = sample_from_sh(shcoeff, lmax=args.num_order, theta=theta, phi=phi)
    
    if mask is not None:
        shading = shading * mask[...,None]
    
    shading = np.float32(shading)
    try:
        ezexr.imwrite(output_path, shading)
        os.chmod(output_path, 0o777)
    except: 
        pass
    
    if args.use_viz==1:
        vizmax_dir = os.path.join(args.output_dir, scene_name, args.vizmax_dir) 
        vizldr_dir = os.path.join(args.output_dir, scene_name, args.vizldr_dir) 
        
        try:
            os.makedirs(vizmax_dir, exist_ok=True)
            os.chmod(vizmax_dir, 0o777)
            os.makedirs(vizldr_dir, exist_ok=True)
            os.chmod(vizldr_dir, 0o777)
        except:
            pass
        
        tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        shading = np.clip(shading, 0, np.inf)
        tonemapped_shading, _, _ = tonemap(shading)
        vizldr_path = os.path.join(vizldr_dir, filename + '.png')
        try:
            skimage.io.imsave(vizldr_path, skimage.img_as_ubyte(tonemapped_shading))
            os.chmod(vizldr_path, 0o777)
        except:
            pass
        
        percentile99 = np.percentile(shading, 99)
        shading_norm = shading / percentile99
        shading_norm = np.clip(shading_norm, 0, 1)
        vizmax_path = os.path.join(vizmax_dir, filename + '.png')
        try:
            skimage.io.imsave(vizmax_path, skimage.img_as_ubyte(shading_norm))
            os.chmod(vizmax_path, 0o777)
        except:
            pass
    return None

# def efficient_rendering(args):
#     queues = []
#     # seek file 
#     print("seeking file...")
#     scenes = os.listdir(args.input_dir)
#     #scenes = ['everett_dining1']
#     for scene_name in tqdm(scenes):
#         input_dir = os.path.join(args.input_dir, scene_name, args.coeff_dir)
#         avalible_files = sorted([a.replace('.npy','') for a in os.listdir(input_dir)])
#         for fname in avalible_files:
#             queues.append(
#                 [scene_name, fname]
#             )
#     queues = queues[args.idx::args.total]
#     fn = partial(process_scene, args)
#     print("Predicting..")
#     with Pool(args.threads) as pool:
#         list(tqdm(pool.imap_unordered(fn, queues), total=len(queues)))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process index and total.")
    parser.add_argument('-i', '--idx', type=int, default=0, help='Index of the item')
    parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')
    parser.add_argument( '--image_width', type=int, default=512, help='size of image to generate in width')
    parser.add_argument( '--image_height', type=int, default=512, help='size of image to generate in height')
    parser.add_argument( '--fov_width', type=int, default=512, help='image size when calcurating fov')
    parser.add_argument('--input_dir', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate", help='input dir')
    parser.add_argument('--output_dir', type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate", help='output dir')
    parser.add_argument('--focal_dir', type=str, default="focal", help='template for coeff dir')
    parser.add_argument('--coeff_dir', type=str, default="shcoeff_perspective_v3_order100", help='template for coeff dir')
    parser.add_argument('--normal_dir', type=str, default="normal", help='template for normal dir')
    parser.add_argument('--shading_dir', type=str, default="shading_exr_perspective_v3_order100_fixinv_ball", help='template for output dir')
    parser.add_argument('--vizmax_dir', type=str, default="shading_exr_perspective_v3_order100_fixinv_ball_viz_max", help='template for vizmax dir')
    parser.add_argument('--vizldr_dir', type=str, default="shading_exr_perspective_v3_order100_fixinv_ball_viz_ldr", help='template for vizldr dir')
    parser.add_argument('--num_order', type=int, default=6)
    parser.add_argument('--apply_integrate', type=int, default=1)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--use_viz', type=int, default=1)
    parser.add_argument('--use_lotus', type=int, default=0)
    parser.add_argument('--use_rotate_normal', type=int, default=0)
    parser.add_argument('--use_ball', type=int, default=1, help="use ball as a normal map")
    parser.add_argument('--use_single_ray_trace', type=int, default=0, help="when use single ray tracing")
    
    args = parser.parse_args()
    #efficient_rendering(args)
    process_scene(args, ['everett_kitchen2', 'dir_30_mip2'])

