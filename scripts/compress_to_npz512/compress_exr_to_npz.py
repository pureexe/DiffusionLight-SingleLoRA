import json 
import ezexr 
from skimage.transform import resize
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
import os

# INDEX_PATH = "/pure/t1/datasets/laion-shading/v4/train/index/150k_marigold.json"
# INPUT_DIR = "/pure/t1/output/DiffusionLight-SingleLoRA/laion-aesthetics-1024"
# OUTPUT_DIR = "/data2/pakkapon/build_shading"
# INPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2"
# OUTPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2_512"

INDEX_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/index/multi_potion_train/multi_all_scenes_train.json"
#INPUT_DIR = "/pure/t1/output/DiffusionLight-SingleLoRA/multi_illumination/real/train"
INPUT_DIR = "/pure/t1/output/DiffusionLight-SingleLoRA/multi_illumination/least_square/train"
OUTPUT_DIR = "/data2/pakkapon/build_shading_least_square"
INPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2"
OUTPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2_512"


def compress_exr_to_npz(meta):
    new_meta = meta.split("/")
    scene, filename = new_meta[0], new_meta[1]
    exr_path = os.path.join(INPUT_DIR, scene, INPUT_NAME, filename+".exr")
    #npz_path = os.path.join(OUTPUT_DIR, scene, OUTPUT_NAME, filename+".npz")
    npz_path = os.path.join(OUTPUT_DIR, scene, filename+".npz")
    if os.path.exists(npz_path):
        return None
    os.makedirs(os.path.join(OUTPUT_DIR, scene), exist_ok=True)
    
    try:
        image = ezexr.imread(exr_path)
    except Exception as e:
        print(f"Error reading {exr_path}: {e}")
        with open("npz_error_log4.txt", "a") as log_file:
            log_file.write(f"{meta}\n")
        return None
    img_resized = resize(image, (512, 512), order=1, anti_aliasing=True).astype(np.float16)
    np.savez_compressed(npz_path, img_resized)
    return None

def main():
    print("READING IMAGE...")
    # READ FILE INDEX
    with open(INDEX_PATH, "r") as f:
        index = json.load(f) 
        queues = index["image_index"]
    
    with Pool(40) as pool:
        results = list(tqdm(pool.imap_unordered(compress_exr_to_npz, queues), total=len(queues)))

if __name__ == "__main__":
    main()