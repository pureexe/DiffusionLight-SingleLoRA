import json 
import skimage
from skimage.transform import resize
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
import os

#INDEX_PATH = "/pure/t1/datasets/laion-shading/v4/train/index/150k_marigold.json"
#INPUT_DIR = "/pure/t1/datasets/laion-aesthetics-1024/images"
#INPUT_DIR = "/data2/pakkapon/relight/download_subset/images"
#OUTPUT_DIR = "/data2/pakkapon/image_512"
#OUTPUT_DIR = "/pure/t1/datasets/laion-aesthetics-1024/images_512"
NUM_SCENES = 816
#INPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2"
#OUTPUT_NAME = "shading_exr_perspective_v3_order6_marigold_v2_512"

def resize_jpg(meta):
    new_meta = meta.split("/")
    scene, filename = new_meta[0], new_meta[1]
    jpg_path = os.path.join(INPUT_DIR, scene, filename+".jpg")
    os.makedirs(os.path.join(OUTPUT_DIR, scene), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, scene, filename+".jpg")
    try:
        image = skimage.io.imread(jpg_path)
    except Exception as e:
        print(f"Error reading {jpg_path}: {e}")
        with open("jpg_error_log_v2.txt", "a") as log_file:
            log_file.write(f"{meta}\n")
        return None
    image = skimage.img_as_float(image)
    img_resized = resize(image, (512, 512), order=1, anti_aliasing=True)
    img_resized = skimage.img_as_ubyte(img_resized)
    skimage.io.imsave(out_path, img_resized)
    return None

def main():
    print("READING IMAGE...")
    # READ FILE INDEX
    with open(INDEX_PATH, "r") as f:
        index = json.load(f) 
        queues = index["image_index"]
    with Pool(40) as pool:
        results = list(tqdm(pool.imap_unordered(resize_jpg, queues), total=len(queues)))

if __name__ == "__main__":
    main()