import numpy as np 
import skimage
import torch
import time 
import os
import argparse
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import os
try:
    import relighting.ezexr as ezexr
except:
    pass


# N = 2(R ⋅ I)R - I

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, required=True ,help='directory that contain the image')
    parser.add_argument("--envmap_dir", type=str, required=True ,help='directory to output environment map') #dataset name or directory
    parser.add_argument("--ball_size", type=int, default=256, help="size of the ball in pixel (width)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--threads", type=int, default=8, help="num thread for pararell processing")
    return parser

def get_reflection_vector_map(I: np.array, N: np.array):
    """
    UNIT-TESTED
    Args:
        I (np.array): Incoming light direction #[None,None,3]
        N (np.array): Normal map #[H,W,3]
    @return
        R (np.array): Reflection vector map #[H,W,3]
    """
    
    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = (I[...,None,:] @ N[...,None])[...,0]
    R = I - 2 * dot_product * N
    return R

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

def get_ideal_normal_ball(size):
    
    """
    UNIT-TESTED
    BLENDER CONVENTION
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

def process_image(args, file_name):
    normal_ball, _ = get_ideal_normal_ball(1024)
    _, mask = get_ideal_normal_ball(256)

    # verify that x of normal is in range [0,1]
    assert normal_ball[:,:,0].min() >= 0 
    assert normal_ball[:,:,0].max() <= 1 
    
    # camera is pointing to the ball, assume that camera is othographic as it placing far-away from the ball
    I = np.array([1, 0, 0]) 
        
    ball_image = np.zeros_like(normal_ball)
    
    # read environment map 
    env_path = os.path.join(args.envmap_dir, file_name)
    env_map = skimage.io.imread(env_path)
    env_map = skimage.img_as_float(env_map)[...,:3]

    env_path = os.path.join(args.envmap_dir, file_name)    
    if file_name.endswith(".exr"):
        env_map = ezexr.imread(env_path)[...,:3].astype(np.float32)
    else:
        env_map = skimage.io.imread(env_path)
        env_map = skimage.img_as_float(env_map)[...,:3]

    
    
    reflected_rays = get_reflection_vector_map(I[None,None], normal_ball)
    spherical_coords = cartesian_to_spherical(reflected_rays)
    
    theta_phi = spherical_coords[...,1:]
    
    # scale to [0, 1]
    # theta is in range [-pi, pi],
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi] 
    theta_phi[...,1] = theta_phi[...,1] / np.pi
    
    # mirror environment map because it from inside to outside
    theta_phi = 1.0 - theta_phi
    
    with torch.no_grad():
        # convert to torch to use grid_sample
        theta_phi = torch.from_numpy(theta_phi[None])
        env_map = torch.from_numpy(env_map[None]).permute(0,3,1,2)
        # grid sample use [-1,1] range
        grid = (theta_phi * 2.0) - 1.0
        ball_image = torch.nn.functional.grid_sample(env_map.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=True)
        ball_image = ball_image[0].permute(1,2,0).numpy()
        ball_image = np.clip(ball_image, 0, 1)
        ball_image = skimage.transform.resize(ball_image, (ball_image.shape[0] // args.scale, ball_image.shape[1] // args.scale), anti_aliasing=True)
        ball_image[~mask] = np.array([0,0,0])
        if file_name.endswith(".exr"):
            ezexr.imwrite(os.path.join(args.ball_dir, file_name), ball_image.astype(np.float32))
        else:
            ball_image = skimage.img_as_ubyte(ball_image)
            skimage.io.imsave(os.path.join(args.ball_dir, file_name), ball_image)

def main():
    # running time measuring
    start_time = time.time()        

    # load arguments
    args = create_argparser().parse_args()
    
    # make output directory if not exist
    os.makedirs(args.ball_dir, exist_ok=True)
    
    # get all file in the directory
    files = sorted(os.listdir(args.envmap_dir))
    
    # create partial function for pararell processing
    process_func = partial(process_image, args)
    
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)  

    
    
    
        
if __name__ == "__main__":
    main()    