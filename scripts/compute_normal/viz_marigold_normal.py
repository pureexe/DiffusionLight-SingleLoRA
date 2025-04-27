import torch
import numpy as np 
from tqdm.auto import tqdm 
import os
import skimage 

SOURCE_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal"
TARGET_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_marigold_viz_x_left"

def from_y_up_to_z_up(normal_map: np.ndarray) -> np.ndarray:
    """
    Convert a normal map from (X-right, Y-up, Z-forward) to (X-forward, Y-right, Z-up).
    
    Args:
        normal_map (np.ndarray): Input normal map with shape [H, W, 3] in range [-1, 1].
    
    Returns:
        np.ndarray: Transformed normal map with the new coordinate system.
    """
    assert normal_map.shape[-1] == 3 # only support catesian coordinate 
    # Define the rotation matrix
    rotation_matrix = np.array([
        [0, 0, 1],  # Z (forward) -> X (forward)
        [1, 0, 0],  # X (right) -> Y (right)
        [0, 1, 0]   # Y (up) -> Z (up)
    ])
    # Apply transformation using broadcasting
    transformed = (rotation_matrix[None, None] @ normal_map[..., None])[..., 0]
        
    return transformed

def from_y_up_to_x_left(normal_map: np.ndarray) -> np.ndarray:
    """
    Convert a normal map from (X-right, Y-up, Z-forward) to (x-left, y-up, Z-forward).
    
    Args:
        normal_map (np.ndarray): Input normal map with shape [H, W, 3] in range [-1, 1].
    
    Returns:
        np.ndarray: Transformed normal map with the new coordinate system.
    """
    assert normal_map.shape[-1] == 3 # only support catesian coordinate 
    # Define the rotation matrix
    rotation_matrix = np.array([
        [-1, 0, 0],  # x (right) -> X (left) Flip X axis
        [0, 1, 0],  # X (up) -> Y (up)
        [0, 0, 1]   # Y (forward) -> Z (forward)
    ])
    # Apply transformation using broadcasting
    transformed = (rotation_matrix[None, None] @ normal_map[..., None])[..., 0]
        
    return transformed

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
            #normal = from_y_up_to_z_up(normal)
            normal = from_y_up_to_x_left(normal)
            normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = (normal + 1) / 2
            normal = np.clip(normal, 0, 1)
            normal = skimage.img_as_ubyte(normal)
            skimage.io.imsave(target_path, normal)
            os.chmod(target_path, 0o777)
        except Exception as e:
            continue
    
    
if __name__ == "__main__":
    main()