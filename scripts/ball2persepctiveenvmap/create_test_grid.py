print("loading libraries...")
import skimage 
import numpy as np
import torch 
print("loading libraries done.")

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


def main():
    print("Creating test grid...")
    _, mask = get_ideal_normal_ball(256)
    
    H, W = 256, 256  # example dimensions

    # Channel 0: horizontal linspace (0 to 1) across width
    x = np.linspace(0, 1, W)

    # Channel 1: vertical linspace (0 to 1) across height
    y = np.linspace(0, 1, H)

    # Use meshgrid to get the full [H,W] grid for each coordinate
    xx, yy = np.meshgrid(x, y)

    # Channel 2: constant 0.5
    z = np.full((H, W), 0.5)

    # Stack along the third dimension to get shape [H, W, 3]
    result = np.stack((xx, yy, z), axis=-1)
    
    result = result * mask[..., None]
    
    result = skimage.img_as_ubyte(result)
    skimage.io.imsave('test_grid.png', result)
    print("test_grid.png saved successfully.")
    
    

    
if __name__ == "__main__":
    main()