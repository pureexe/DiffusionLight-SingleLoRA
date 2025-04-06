# reverse wrapping is for chaning from chromeball to environment map 

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

def create_envmap_grid(size: int):
    """
    PYSHTOOL CONVENTION (x-forward, y-right, z-up)
    Create the grid of environment map that contain the position in sperical coordinate
    # Top left is (theta=-0.5,phi=-1) and bottom right is (theta=0.5, phi=1)
    Top left is (theta=-pi/2,phi=-pi) and bottom right is (theta=pi/2, phi=pi)
    """    
    
    theta = torch.linspace(np.pi / 2, -np.pi / 2, size)
    #phi = torch.linspace(0, 2*np.pi, size * 2) # need to use [0,2pi] to match pyshtool convention
    phi = torch.linspace(-np.pi, np.pi, size * 2) # need to use [0,2pi] to match pyshtool convention

    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='ij')     
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt" ,help='directory that contain the image') 
    parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal",help='directory that contain horizontal focal file.') 
    parser.add_argument("--envmap_dir", type=str, default="output/envmap_grid/14n_copyroom1_180" ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--envmap_height", type=int, default=128, help="size of the environment map height in pixel (height)")
    parser.add_argument("--ball_ratio", type=float, default=128 / 512, help="size of the environment map height in pixel (height)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--fov_width", type=int, default=512, help="size of image to calcurate focal")
    parser.add_argument("--threads", type=int, default=25, help="num thread for pararell processing")
    return parser


# def create_argparser():    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/square_hdr_gt" ,help='directory that contain the image') 
#     parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal",help='directory that contain horizontal focal file.') 
#     parser.add_argument("--envmap_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/envmap" ,help='directory to output environment map') #dataset name or directory 
#     parser.add_argument("--envmap_height", type=int, default=128, help="size of the environment map height in pixel (height)")
#     parser.add_argument("--ball_ratio", type=float, default=128 / 512, help="size of the environment map height in pixel (height)")
#     parser.add_argument("--scale", type=int, default=4, help="scale factor")
#     parser.add_argument("--fov_width", type=int, default=512, help="size of image to calcurate focal")
#     parser.add_argument("--threads", type=int, default=25, help="num thread for pararell processing")
#     return parser

# def create_argparser():    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/input" ,help='directory that contain the image') 
#     parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/focal",help='directory that contain horizontal focal file.') 
#     parser.add_argument("--envmap_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/envmap_grid" ,help='directory to output environment map') #dataset name or directory 
#     parser.add_argument("--envmap_height", type=int, default=128, help="size of the environment map height in pixel (height)")
#     parser.add_argument("--ball_ratio", type=float, default=128 / 512, help="size of the environment map height in pixel (height)")
#     parser.add_argument("--scale", type=int, default=4, help="scale factor")
#     parser.add_argument("--fov_width", type=int, default=512, help="size of image to calcurate focal")
#     parser.add_argument("--threads", type=int, default=25, help="num thread for pararell processing")
#     return parser

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


def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

def get_rand_normal_vector():
    """
    random normal vector to be initialize
    """
    rand_value = np.random.uniform(low=-1, high=1, size=(2,))
    x_value = np.sqrt(np.clip(1 - (rand_value[0]**2) - (rand_value[1]**2),0, np.inf))
    return x_value, rand_value[0], rand_value[1]

def solve_for_normal(reflect_directions, ball_distance):
    """
    Using scipy's least square to solve for normal direction
    @params
        - reflect_direction (np.darray) : reflect vector direction from environment map [H,W,3] in format of [x-forward, y-right, z-up]
        - ball_distance (float) : distance to place a unit chromeball
    @returns
        np.darray: normal_directions  [H,W,3] in format of [x-forward, y-right, z-up]
    """
    normal_directions = np.zeros_like(reflect_directions) #H,W,3
    
    # def make_equations(d, Lx, Ly, Lz):
    #     """ 
    #     Input distance (d) and Reflection vector (L) in 3 direction
    #     Returns a function that computes the equations for given (x, y, z)
    #     """
    #     def equations(variables):
    #         """
    #         Assume the camera is at [0,0,0] the chromeball unit size 1 at [-d,0,0] (we need -d to preserve pyshtool convention)
    #         """
    #         x, y, z = variables  # Unknowns
    #         S = (x + d) * Lx + y * Ly + z * Lz  # Compute S
    #         denom = np.sqrt(np.clip(x**2 + y**2 + z**2, 0, np.inf))  # Common denominator
    #         return [
    #             2 * S * (x + d) - Lx + (x / denom), #normal in x axis from given reflect-vector L and assume camera at origin
    #             2 * S * y - Ly + (y / denom), # normal in y axis
    #             2 * S * z - Lz + (z / denom), # normal in z axis
    #             (x + d) ** 2 + y ** 2 + z ** 2 - 1,  # Added constraint that normal should be on sphere surface 
    #             #np.clip((x + d), 0, np.inf) # Enforce x to be negative?
    #         ]
    #     return equations

    def make_equations(d, Lx, Ly, Lz):
        """ 
        Input distance (d) and Reflection vector (L) in 3 direction
        Returns a function that computes the equations for given (x, y, z)
        """
        def equations(variables):
            """
            Assume the camera is at [0,0,0] the chromeball unit size 1 at [-d,0,0] (we need -d to preserve pyshtool convention)
            """
            x, y, z = variables  # Unknowns
            # force x to always be positive
            # scale from [-1,1] to [0,1]
            S = (x) * Lx + y * Ly + z * Lz  # Compute S
            denom = np.sqrt(np.clip((x-d)**2 + y**2 + z**2, 0, np.inf))  # Common denominator
            return [
                2 * S * (x) - Lx + ((x-d) / denom), #normal in x axis from given reflect-vector L and assume camera at origin
                2 * S * y - Ly + (y / denom), # normal in y axis
                2 * S * z - Lz + (z / denom), # normal in z axis
                (x) ** 2 + y ** 2 + z ** 2 - 1,  # Added constraint that normal should be on sphere surface 
            ]
        return equations

    
    mask = np.zeros_like(reflect_directions[...,0])
    for i in tqdm(range(reflect_directions.shape[0])):
        for j in range(reflect_directions.shape[1]):
            Lx, Ly, Lz = reflect_directions[i, j]

            # Set up and solve least squares for this pixel
            equations_func = make_equations(ball_distance, Lx, Ly, Lz)
            initial_guess = [1, 0, 0]
            
            result = least_squares(
                equations_func, 
                initial_guess,
                bounds=([0, -1, -1], [1, 1, 1]), # bounds for x,y,z
            )
            
            # Extract and normalize result
            x, y, z = result.x
            #x = (x + 1) / 2 # rescale to match the enforce range
            normal_vec = np.array([ x, y, z])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            mask[i,j] =  x >= 0 

            # Store result
            normal_directions[i, j] = normal_vec

    return normal_directions, mask
    


def process_image(args: argparse.Namespace, file_name: str):
    # check if exist, skip!
    envmap_output_path = os.path.join(args.envmap_dir, file_name)
    # if os.path.exists(envmap_output_path):
    #     return None

    # read ball image 
    ball_path = os.path.join(args.ball_dir, file_name)
    try:
        if file_name.endswith(".exr"):
            ball_image = ezexr.imread(ball_path)
        else:
            ball_image = skimage.io.imread(ball_path)
            ball_image = skimage.img_as_float(ball_image)
    except:
        #print("FAILED TO READ BBALL")
        return None # failed to read image
        
    # read focal length
    try:
        fov = get_fov(args, file_name) # 36.7811506859767
        # print("FOV: ---> ", fov * 180 / np.pi)
    except:
        print("FAILED TO READ FOV")
        return None
    
    # get reflect vector for environment map
    env_grid  = create_envmap_grid(args.envmap_height )   # * args.scale # [phi [0,2pi], theta [pi/2, -pi/2]]
    theta, phi = env_grid[...,0], env_grid[...,1]
    reflect_directions = get_cartesian_from_spherical(theta, phi) # (x-forward, y-right, z-up) range [-1,1]
    half_angle = get_chromeball_half_angle(fov,args.ball_ratio)
    ball_distance = get_chromeball_distance(half_angle)
    normal_directions, mask = solve_for_normal(reflect_directions, ball_distance)
    
    # save normal map
    normal_map = (normal_directions + 1.0) / 2.0
    normal_map = skimage.img_as_ubyte(normal_directions)
    skimage.io.imsave(envmap_output_path+'.normal.png', normal_map)
    
    # We ignore X axis because it represent forward. (y-right, z-up) range [-1,1]
    y,z = normal_directions[...,1], normal_directions[...,2] # we use normal too indicate location on the chromeball.
    pos = np.concatenate([y[...,None],z[...,None]], axis=-1)
    
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
        ball_image = torch.from_numpy(ball_image[None]).float()
        ball_image = ball_image.permute(0,3,1,2) # [1,3,H,W]
        
        env_map = torch.nn.functional.grid_sample(ball_image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        env_map = env_map[0].permute(1,2,0).numpy()
        
    #env_map = env_map * mask[...,None] + (np.ones_like(env_map) * (1 - mask[..., None])) # apply mask to envmap

    # TODO: apply mask for in-visible region
    env_map_default = skimage.transform.resize(env_map, (args.envmap_height, args.envmap_height*2), anti_aliasing=True)
    if file_name.endswith(".exr"):
        ezexr.imwrite(envmap_output_path, env_map_default.astype(np.float32))
        if VIZ_TONEMAP:
            tonemap = TonemapHDR(2.4,99,0.9)
            image, _, _ = tonemap(env_map_default)
            image = skimage.img_as_ubyte(image)
            skimage.io.imsave(envmap_output_path+'.png', image)
    else:
        env_map_default = skimage.img_as_ubyte(env_map_default)        
        skimage.io.imsave(envmap_output_path, env_map_default)
    os.chmod(envmap_output_path, 0o777)
    return None

    


def main(args):
    # running time measuring
    start_time = time.time()        
    
    # make output directory if not exist
    os.makedirs(args.envmap_dir, exist_ok=True)
    os.chmod(args.envmap_dir, 0o777)
    
    # get all file in the directory
    files = sorted(os.listdir(args.ball_dir))

    # create partial function for pararell processing
    process_func = partial(process_image, args)
    
    process_func(files[2])
    exit()
    
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)        


    
if __name__ == "__main__":
    # load arguments
    args = create_argparser().parse_args()
    main(args)