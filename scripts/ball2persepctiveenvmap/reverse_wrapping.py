# reverse wrapping is for chaning from chromeball to environment map 

print("LOADING LIBRARY")
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
from tonemapper import TonemapHDR
from scipy.optimize import root
from scipy.optimize import least_squares

import math 

try:
    import ezexr
except:
    pass

print("LIBRARY DONE")

VIZ_TONEMAP = True
USE_PYTHON_RENDER = True

def create_envmap_grid(size: int):
    """
    PYSHTOOL CONVENTION (x-forward, y-right, z-up)
    Create the grid of environment map that contain the position in sperical coordinate
    We need a proper supprot for the align_corner
    # Top left is (theta=-0.5,phi=-1) and bottom right is (theta=0.5, phi=1)
    Top left is (theta=-pi/2,phi=-pi) and bottom right is (theta=pi/2, phi=pi)
    """    
    # create theta
    theta = torch.arange(0, size) # [0,size-1] # this value will act as a middle pixel is on the edge, but in the environment map we need corner to be the edge
    theta = (theta + 0.5) / size # [0.5 / size, size - 0.5 / size]
    # rescale from [0,1] to [-pi/2, pi/2]
    theta = theta * np.pi - (np.pi / 2) # [0,1] to [-pi/2, pi/2]
    # flip the value from [-pi/2,pi/2] to [pi/2,-pi/2]
    theta = torch.flip(theta, dims=[0]) # flip the value from [0,1] to [1,0]   
    
    # create phi
    phi = torch.arange(0, size * 2) # [0,size-1] # this value will act as a middle pixel is on the edge, but in the environment map we need corner to be the edge
    phi = (phi + 0.5) / (size * 2) # [0.5 / (size * 2), (size * 2) - 0.5 / (size * 2)]
    
    if USE_PYTHON_RENDER:
        # python code / blender convention use -pi to pi
        # rescale from [0,1] to [-pi, pi]
        phi = phi * (2 * np.pi) - np.pi # [0,1] to [-pi, pi]
    else:
        # pyshtool convention use 0 to 2pi
        # rescale from [0,1] to [0,2pi]
        phi = phi * (2 * np.pi) # [0,1] to [0,2pi]

    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='ij')     
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/input_copyroom1" ,help='directory that contain the image') 
    parser.add_argument("--focal_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal",help='directory that contain horizontal focal file.') 
    parser.add_argument("--envmap_dir", type=str, default="output/envmap_grid/14n_copyroom1_256s4_v1" ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--envmap_height", type=int, default=256, help="size of the environment map height in pixel (height)")
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
#     parser.add_argument("--envmap_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/envmap_grid/ball_grid" ,help='directory to output environment map') #dataset name or directory 
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

def save_normal(envmap_output_path, normal_directions):
    # save normal map
    if normal_directions.shape[-1] == 2:
        # convert to spherical coordinate
        normal_directions = cartesian_to_spherical(normal_directions[...,0], normal_directions[...,1] )
    normal_map = (normal_directions + 1.0) / 2.0
    normal_map = skimage.img_as_ubyte(normal_map)
    skimage.io.imsave(envmap_output_path+'.normal.png', normal_map)

def spherical_to_cartesian(theta, phi):
    """
    Converts spherical coordinates (theta, phi) to unit vectors.

    Parameters:
    theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
    phi (numpy.ndarray): Array of phi values in the range [0, 2*pi]. # compatible with [-pi, pi]

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

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    x-forward, y-right, z-uup
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

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    #N = 2(R ⋅ I)R - I
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N

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

# def solve_for_normal(reflect_directions, ball_distance, half_ball_angle):
#     """
#     Using scipy's least square to solve for normal direction
#     @params
#         - reflect_direction (np.darray) : reflect vector direction from environment map [H,W,3] in format of [x-forward, y-right, z-up]
#         - ball_distance (float) : distance to place a unit chromeball
#     @returns
#         np.darray: normal_directions  [H,W,3] in format of [x-forward, y-right, z-up]
#     """
#     #normal_directions = np.zeros_like(reflect_directions) #H,W,3
#     H = reflect_directions.shape[0]
#     W = reflect_directions.shape[1]
#     normal_directions = np.zeros((H,W,2)) #H,W,2
    
#     def make_equations(d, Lx, Ly, Lz):
#         """ 
#         Input distance (d) and Reflection vector (L) in 3 direction
#         Returns a function that computes the equations for least square
#         @see https://imgur.com/a/2eIaoBb
#         """
#         assert d >= 0 # distance (d) is garuntee to be positive.
        
#         def equations(variables):
#             """
#             Assume the camera is at [0,0,0] the chromeball unit size 1 at [-d,0,0] (we need -d to preserve pyshtool convention)
#             """
#             theta, phi = variables  # Unknowns
#             x, y, z = spherical_to_cartesian(theta, phi) # convert to cartesian
#             # this x is garuntee to be in [-1,1]. In powerpoint, x can be -9.5 when d is -10. so, we need to shift x value from center at 0 to center at d
#             x = (-d) + x # (-d) is the center of chromeball. 
            
#             S = (x+d) * Lx + y * Ly + z * Lz  # Compute S
#             denom = np.sqrt(np.clip(x**2 + y**2 + z**2, np.finfo(np.float32).eps, np.inf))  # Common denominator

#             return [
#                 (2 * S * (x+d)) - Lx + (x / denom), #normal in x axis from given reflect-vector L and assume camera at origin
#                 (2 * S * y) - Ly + (y / denom), # normal in y axis
#                 (2 * S * z) - Lz + (z / denom), # normal in z axis
#                 ((x+d) ** 2 + y ** 2 + z ** 2) - 1,  # (no longer needed, theta and phi are already constrained on the surface) # Added constraint that normal should be on sphere surface 
#             ]
#         return equations

#     # compute intial normal from orthographic projection
#     for i in tqdm(range(reflect_directions.shape[0])):
#         for j in range(reflect_directions.shape[1]):
#             Lx, Ly, Lz = reflect_directions[i, j]

#             # Set up and solve least squares for this pixel
#             equations_func = make_equations(ball_distance, Lx, Ly, Lz)
#             initial_guess = [0, 0]
#             result = least_squares(
#                 equations_func, 
#                 initial_guess,
#                 bounds=([
#                         -np.pi/2, # theta (vertical, lower bound)
#                         -np.pi/2 # phi (horizontal, lower bound)
#                     ],[
#                         np.pi/2, # theta (vertical, upper bound)
#                         np.pi/2 # phi (horizontal, upper bound)
#                 ]), # bounds to be front side only
#                 verbose=0,
#                 # xtol=1e-12,  # tolerance for solution vector
#                 # ftol=None,  # tolerance for cost function
#                 # gtol=None,  # tolerance for gradient
#                 # max_nfev=10000  # increase if needed
#             )
#             # Extract and normalize result
#             theta, phi = result.x

#             # Store result
#             normal_directions[i, j, 0] = theta
#             normal_directions[i, j, 1] = phi

#     return normal_directions


def solve_for_normal(reflect_directions, ball_distance, half_ball_angle):
    """
    Using scipy's least square to solve for normal direction
    @params
        - reflect_direction (np.darray) : reflect vector direction from environment map [H,W,3] in format of [x-forward, y-right, z-up]
        - ball_distance (float) : distance to place a unit chromeball
    @returns
        np.darray: normal_directions  [H,W,3] in format of [x-forward, y-right, z-up]
    """
    H = reflect_directions.shape[0]
    W = reflect_directions.shape[1]
    normal_directions = np.zeros((H,W,3)) #H,W,3
    
    def make_equations(d, Lx, Ly, Lz):
        """ 
        Input distance (d) and Reflection vector (L) in 3 direction
        Returns a function that computes the equations for least square
        @see https://imgur.com/a/2eIaoBb
        """
        assert d >= 0 # distance (d) is garuntee to be positive.
        
        def equations(variables):
            """
            Assume the camera is at [0,0,0] the chromeball unit size 1 at [-d,0,0] (we need -d to preserve pyshtool convention)
            """
            
            y,z = variables  # Unknowns
            x_square = 1 - (y**2) - (z**2)
            x = np.sqrt(np.clip(x_square,0, np.inf)) # assume that we only optimized for front side of chromeball
           
            # this x is garuntee to be in [0,1]. In powerpoint, x can be -9.5 when d is -10. so, we need to shift x value from center at 0 to center at d
            x = (-d) + x # (-d) is the center of chromeball. 
 
            S = (x+d) * Lx + y * Ly + z * Lz  # Compute S
            denom = np.sqrt(np.clip(x**2 + y**2 + z**2, np.finfo(np.float32).eps, np.inf))  # Common denominator

            return [
                (2 * S * (x+d)) - Lx + (x / denom), #normal in x axis from given reflect-vector L and assume camera at origin
                (2 * S * y) - Ly + (y / denom), # normal in y axis
                (2 * S * z) - Lz + (z / denom), # normal in z axis
                ((x+d) ** 2 + y ** 2 + z ** 2) - 1,  # Added constraint that normal should be on sphere surface 
                np.maximum(-x_square, 0) # this is to make sure that x is positive
            ]
        return equations

    # compute intial normal from orthographic projection
    for i in tqdm(range(reflect_directions.shape[0])):
        for j in range(reflect_directions.shape[1]):
            Lx, Ly, Lz = reflect_directions[i, j]

            # Set up and solve least squares for this pixel
            equations_func = make_equations(ball_distance, Lx, Ly, Lz)
            initial_guess = [0, 0]
            result = least_squares(
                equations_func, 
                initial_guess,
                bounds=([
                        -1, # y (horizontal, lower bound)
                        -1 # z (vertical, lower bound)
                    ],[
                        1, # y (horizontal, upper bound)
                        1 # z (vertical, upper bound)
                ]), # bounds to be front side only
                verbose=0,
                # xtol=1e-12,  # tolerance for solution vector
                # ftol=None,  # tolerance for cost function
                # gtol=None,  # tolerance for gradient
                # max_nfev=10000  # increase if needed
            )
            # Extract and normalize result
            y, z = result.x
            x = np.sqrt(np.clip(1 - (y**2) - (z**2),0, np.inf))
            normal_directions[i, j, 0] = x
            normal_directions[i, j, 1] = y
            normal_directions[i, j, 2] = z

    return normal_directions


def get_coordinates_from_normal(normals, half_angle, ball_distance):
    # convert from spherical coordinate to cartesian coordinate if needed
    if normals.shape[-1] == 2:
        normals = spherical_to_cartesian(normals[...,0], normal_spherical[...,1])
    
    x,y,z = normals[...,0], normals[...,1], normals[...,2]
    
    # x is now in [-1,1] but we need to shift it to ball center at -d
    x = (-ball_distance) + x # (-d) is the center of chromeball.
    
    f = 1 / math.tan(half_angle) # f is the focal length

    # since all point is in -x and camera looking to -z. the camera plane  at focal length (f) will be at -f
    u = y * -f / x # [H,W]
    v = z * -f / x # [H,W]
    
    pos = np.stack([u, v], axis=-1) # [H,W,2] (y-right, z-up)    
        
    return pos 


def process_image(args: argparse.Namespace, file_name: str):
    # check if exist, skip!
    envmap_output_path = os.path.join(args.envmap_dir, file_name)
    # TODO: TEMPORARY DISTABLE, will reenable after fixing the bug
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
    env_grid  = create_envmap_grid(args.envmap_height * args.scale)   # * args.scale # [phi [0,2pi], theta [pi/2, -pi/2]]
    theta, phi = env_grid[...,0], env_grid[...,1]
    reflect_directions = get_cartesian_from_spherical(theta, phi) # (x-forward, y-right, z-up) range [-1,1]
    
    # compute the angle
    half_angle = get_chromeball_half_angle(fov,args.ball_ratio)
    half_ball_angle = np.pi/2 - half_angle    
    ball_distance = get_chromeball_distance(half_angle)

    # solving for normal
    normal_directions = solve_for_normal(reflect_directions, ball_distance, half_ball_angle)
    save_normal(envmap_output_path, normal_directions)
    
    # get coordinate to sample from normal 
    pos = get_coordinates_from_normal(normal_directions, half_angle, ball_distance) # [H,W,2] (y-right, z-up)
    
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
    
    # save image
    env_map_default = skimage.transform.resize(env_map, (args.envmap_height, args.envmap_height*2), anti_aliasing=True)
    env_map_default = env_map
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
    
    process_func(files[0])
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