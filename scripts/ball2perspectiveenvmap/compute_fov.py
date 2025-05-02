import os 
import numpy as np


focal_path = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen6/focal/dir_0_mip2.npy"
fov_width = 512
focal_px = np.load(focal_path) # focal length in term of pxiel
fov_rad = 2 * np.arctan2(fov_width, 2*focal_px)
print("FOV in degree: ")
print(fov_rad * 180 / np.pi)
print('Distance in unit: ')
distance = 1 / np.sin(fov_rad / 2)
print(distance)