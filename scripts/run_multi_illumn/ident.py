import os
import numpy as np

focal_dir = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/train/joy_bedroom18/focal"

for fname in os.listdir(focal_dir):
    if fname.endswith('.npy'):
        path = os.path.join(focal_dir, fname)
        try:
            np.load(path)
        except Exception as e:
            print(f"Error in {path}: {e}")