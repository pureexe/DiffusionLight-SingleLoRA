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

VIZ_TONEMAP = True

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_reverse_unwrap_ball_inverse2" ,help='directory that contain the image') 
    parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal",help='directory that contain horizontal focal file.') 
    parser.add_argument("--envmap_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/envmap_reverse_unwrap" ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--ball_size", type=int, default=256, help="size of the environment map height in pixel (height)")
    parser.add_argument("--ball_ratio", type=float, default=128 / 512, help="size of the environment map height in pixel (height)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--fov_width", type=int, default=512, help="size of image to calcurate focal")
    parser.add_argument("--threads", type=int, default=25, help="num thread for pararell processing")
    return parser

def get_ideal_normal_ball(size):
    """
    convention
    X: forward
    Y: right
    Z: up
    
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(1, -1, size)
    
    #use indexing 'xy' torch match vision's homework 3
    y,z = torch.meshgrid(y, z ,indexing='xy') 
    
    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    x = torch.sqrt(x)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def get_ball_theta_phi(size, azimuth_limit=np.pi, elevation_limit=np.pi/2):
    """
    Create a normal map using azimuth and elevation angles.

    X: forward
    Y: right
    Z: up

    @params
        - size (int): resolution (height and width)
        - azimuth_limit (float): max absolute azimuth angle (default: pi)
        - elevation_limit (float): max absolute elevation angle (default: pi/2)
    @returns
        - theta_phi
    """
    # Create azimuth and elevation ranges
    # Now value is in range [-pi, pi] in azimute
    azimuth = torch.linspace(-azimuth_limit, azimuth_limit, size)     # horizontal (Y-axis)
    elevation = torch.linspace(elevation_limit, -elevation_limit, size)  # vertical (Z-axis)

    # convert azimuth to [0, 2*pu]
    azimuth = (azimuth + 2 * np.pi) % (2 * np.pi)


    azimuth, elevation = torch.meshgrid(azimuth, elevation, indexing='xy')
    
    theta_phi = torch.stack([azimuth, elevation], dim=-1)  # [size, size, 2]
    theta_phi = theta_phi.numpy()
    return theta_phi


def create_ndc_grid(ball_size: int,ball_ratio: float):
    """
    Create NDC coordinate  (U,V) for the chromeball. 
    U is horizontal, V is vertical. U-right, V-up
    """
    u = torch.linspace(-ball_ratio, ball_ratio, ball_size)
    v = torch.linspace(ball_ratio, -ball_ratio, ball_size) # need to use [0,2pi] to match pyshtool convention


    #use indexing 'xy' torch match vision's homework 3
    u, v = torch.meshgrid(u, v ,indexing='xy')     
    
    pos_ndc= torch.cat([u[..., None], v[..., None]], dim=-1)
    pos_ndc = pos_ndc.numpy()
    return pos_ndc

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    
def get_viewing_directions(ball_size: int,ball_ratio: float, fov: float):
    """
    Create camera coordinate
    Convention PYSHTOOL (x-forward, y-right, z-up)
    Ball is placing at (-d, 0, 0) camera is at (0,0,0)
    @see https://imgur.com/a/G2ythUM
    """
    pos_ndc = create_ndc_grid(ball_size, ball_ratio)
    half_tan = math.tan(fov / 2)    
    pos = pos_ndc * half_tan
    viewing_directions = np.ones((ball_size,ball_size,3))
    viewing_directions[...,0] *= 1 # we look into -x 
    viewing_directions[...,1:3] = pos
    viewing_directions = normalize(viewing_directions)
    return viewing_directions

def spherical_to_cartesian(theta, phi):
    """
    Converts spherical coordinates (theta, phi) to unit vectors.

    Parameters:
    theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
    phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].

    Returns:
    numpy.ndarray: Output array of shape (..., 3), representing unit vectors.
    """
    # Ensure inputs are numpy arrays
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Calculate components of the unit vectors
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    # Stack components into output array
    vectors = np.stack([x, y, z], axis=-1)

    return vectors

def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)

def get_normal_directions(ball_size: int,ball_ratio: float, theta_ball: float):
    pos_ndc = create_ndc_grid(ball_size, ball_ratio)
    pos_ndc = pos_ndc / ball_ratio# scale ndc to [-1,1]
    pos_cam = theta_ball * pos_ndc  # slide 5, get position
    # horizontal is currently in [-pi,pi], convert to [0,2pi]
    pos_cam[...,0] = (pos_cam[...,0] + 2 * np.pi) % (2 * np.pi)
    
    phi = pos_cam[...,0] # horizontal
    theta = pos_cam[...,1] # vertical
    
    pos = spherical_to_cartesian(theta, phi) # convert to cartesian coordinate
    
    # slide 6 convert to from spherical coordinate to cartesian coordinate
    # (x-foward, y-right, z-up) #FAILED TO TEST

    return pos
    
def get_reflection_directions(I: np.ndarray, N: np.ndarray):
    """
    Args:
        I (np.ndarray): viewing direction #[H,W,3]
        N (np.ndarray): Normal map #[H,W,3]
    @return
        R (np.ndarray): Reflection vector map #[H,W,3]
    """
    
    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = np.sum(I * N, axis=-1, keepdims=True)  # shape [H, W, 1]
    R = I - 2 * dot_product * N  # shape [H, W, 3]
    return R

def get_fov(args, filename):
    new_filename = filename.split('.')[0]+'.npy'
    focal_path = os.path.join(args.focal_dir, new_filename)
    focal_px = np.load(focal_path) # focal length in term of pxiel
    fov_rad = 2 * np.arctan2(args.fov_width, 2*focal_px)
    return fov_rad

def get_chromeball_half_angle(fov, ball_ratio):
    # get how much half angle of chromeball over
    # @see https://imgur.com/a/pIeaxmg
    theta = math.atan(math.tan(fov / 2) * ball_ratio)
    return theta

def get_chromeball_distance(theta):
    # Get chroomeball distance (d) assume chromeball has 1 unit radius
    # @see https://imgur.com/DJMBhbI
    inv_theta = np.pi / 2 - theta
    d = (math.sin(inv_theta) / math.tan(theta)) + math.cos(inv_theta)
    return d

def get_theta_ball(fov, ball_ratio):
    theta = get_chromeball_half_angle(fov, ball_ratio)
    theta_ball = np.pi / 2 - theta 
    return theta_ball

def process_image(args: argparse.Namespace, file_name: str):
    # check if exist, skip!
    ball_output_path = os.path.join(args.ball_dir, file_name)
    # if os.path.exists(envmap_output_path):
    #     return None
    
    
    # get fov
    try:
        fov = get_fov(args, file_name)
    except:
        print("FAILED TO READ FOV")
        return None
    
    # get theta ball
    theta_ball = get_theta_ball(fov, args.ball_ratio)    
    
    # get environment map
    env_path = os.path.join(args.envmap_dir, file_name)
    try:
        if file_name.endswith(".exr"):
            envmap_image = ezexr.imread(env_path)
        else:
            envmap_image = skimage.io.imread(env_path)
            envmap_image = skimage.img_as_float(env_path)
    except:
        #print("FAILED TO READ BBALL")
        return None # failed to read image
    
    ideal_normalball, mask = get_ideal_normal_ball(args.ball_size) # i just need a mask for a ball.
    
    # get viewing direction
    viewing_directions = get_viewing_directions(args.ball_size, args.ball_ratio, fov)   

    # fidning normal map
    normal_directions = get_normal_directions(args.ball_size, args.ball_ratio, theta_ball)

    # save normal_direction to png file
    normal_img = normal_directions * 0.5 + 0.5 # convert to [0,1]
    normal_img = np.clip(normal_img, 0, 1)
    normal_img = normal_img * mask[...,None] # apply mask)
    normal_img = skimage.img_as_ubyte(normal_img)
    skimage.io.imsave(ball_output_path+'_normal.png', normal_img)
    
    
    # get reflection direction
    reflection_directions = get_reflection_directions(viewing_directions, normal_directions)

    spherical_coords = cartesian_to_spherical(reflection_directions)
    theta_phi = spherical_coords[...,1:]
    

    # scale to [0, 1]
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi] 
    theta_phi[...,1] = theta_phi[...,1] / np.pi
    
    # scale to [-1,1]
    theta_phi= theta_phi * 2 - 1

    
    pos = theta_phi # remove x axis

    
    # since Z-UP (top 1, bottom -1) but torch grid sample is top -1 bottom 1 (is z-down) we flip only z axis 
    # but if we use for blender rendering, it looking from inside, so it need to flip y axis as well.
    LOOKING_FROM_INSIDE = False 
    if LOOKING_FROM_INSIDE:
        pos  = -pos
    else:
        pos[...,1] = -pos[...,1] 
    
    # using pytorch method for bilinear interpolation
    with torch.no_grad():
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos)[None].float()
        # convert ball to support pytorch
        envmap_image = torch.from_numpy(envmap_image[None]).float()
        envmap_image = envmap_image.permute(0,3,1,2) # [1,3,H,W]
        
        ball_image = torch.nn.functional.grid_sample(envmap_image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        ball_image = ball_image[0].permute(1,2,0).numpy()
        
        # apply mask
        ball_image = ball_image * mask[...,None]

    if file_name.endswith(".exr"):
        ezexr.imwrite(ball_output_path, ball_image.astype(np.float32))
        if VIZ_TONEMAP:
            tonemap = TonemapHDR(2.4,99,0.9)
            image, _, _ = tonemap(ball_image)
            image = skimage.img_as_ubyte(image)
            skimage.io.imsave(ball_output_path+'.png', image)
    else:
        ball_image = skimage.img_as_ubyte(ball_image)        
        skimage.io.imsave(ball_output_path, ball_image)
    os.chmod(ball_output_path, 0o777)
    return None



def main(args):
    # running time measuring
    start_time = time.time()        
    
    # make output directory if not exist
    os.makedirs(args.ball_dir, exist_ok=True)
    os.chmod(args.envmap_dir, 0o777)
    
    # get all file in the directory
    files = sorted(os.listdir(args.envmap_dir))

    # create partial function for pararell processing
    process_func = partial(process_image, args)
    # only process first file for debug first
    process_func(files[0])
    exit()
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)
    
if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)