import os
import tarfile
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import json
import time 

# INDEX_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/index/multi_potion_train/multi_all_scenes_train.json"
# PROMPT_DIR = "/data2/pakkapon/build_prompt"
# IMAGE_DIR = "/data2/pakkapon/image_512"
# SHADING_DIR = "/data2/pakkapon/build_shading_least_square"
# OUTPUT_DIR = "/data2/pakkapon/multi_ilumination_least_square_train"

INDEX_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/index/multi_potion_train/multi_all_scenes_train.json"
PROMPT_DIR = "/data2/pakkapon/build_prompt"
IMAGE_DIR = "/data2/pakkapon/image512_lstsq"
SHADING_DIR = "/data2/pakkapon/build_shading_least_square"
OUTPUT_DIR = "/data2/pakkapon/lstsq_image_lstsq_shading"
samples_per_shard = 1000

# INDEX_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/index/multi_potion_train/multi_all_scenes_train.json"
# PROMPT_DIR = "/data2/pakkapon/build_prompt"
# IMAGE_DIR = "/data2/pakkapon/image512_real"
# SHADING_DIR = "/data2/pakkapon/build_shading_least_square"
# OUTPUT_DIR = "/data2/pakkapon/real_image_lstsq_shading"
# samples_per_shard = 1000



def create_tar(num_samples, num_shards, shard_id):
    start_idx = shard_id * samples_per_shard
    end_idx = min((shard_id + 1) * samples_per_shard, num_samples)
    shard_name = os.path.join(OUTPUT_DIR, f"train-{shard_id:04d}.tar")
    with open(INDEX_PATH, "r") as f:
        index = json.load(f) 
        queues = index["image_index"]

    with tarfile.open(shard_name, "w") as tar:
        for idx in range(start_idx, end_idx):
            base = queues[idx]
            base_split= base.split('/')
            scene_name = base_split[0]
            image_name = base_split[1]
            #filename = base_split[0] + '+' + base_split[1]
            txt_path = os.path.join(PROMPT_DIR, scene_name,  f"{image_name}.txt")
            jpg_path = os.path.join(IMAGE_DIR, scene_name, f"{image_name}.jpg")
            npz_path = os.path.join(SHADING_DIR, scene_name, f"{image_name}.npz")

            tar.add(txt_path, arcname=f"{scene_name}+{image_name}.txt")
            tar.add(jpg_path, arcname=f"{scene_name}+{image_name}.jpg")
            tar.add(npz_path, arcname=f"{scene_name}+{image_name}.npz")

def main():
    print("READING IMAGE...")
    # READ FILE INDEX
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("CREATING TAR")
    start_time = time.time()

    with open(INDEX_PATH, "r") as f:
        index = json.load(f) 
        queues = index["image_index"]

    num_samples = len(queues)
    num_shards = (num_samples + samples_per_shard - 1) // samples_per_shard
    fn = partial(create_tar, num_samples, num_shards)
    with Pool(2) as pool:
        results = list(tqdm(pool.imap_unordered(fn, range(num_shards)), total=num_shards))
    print("TOTAL TIME: ", time.time() - start_time)
    print("DONE")
    
if __name__ == "__main__":
    main()