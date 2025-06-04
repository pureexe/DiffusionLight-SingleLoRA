import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# PROMPT_FILE = "/pure/t1/datasets/laion-shading/v4/train/prompts.json"
# INDEX_FILE = "/pure/t1/datasets/laion-shading/v4/train/index/150k_marigold.json"
# OUTPUT_DIR = "/data2/pakkapon/build_prompt"

PROMPT_FILE = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/prompts.json"
INDEX_FILE = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/index/multi_potion_train/multi_all_scenes_train.json"
OUTPUT_DIR = "/data2/pakkapon/build_prompt"

NUM_PROCESSES = 32 

def save_prompt(item):
    k, prompt = item
    scene, name = k.split("/")
    output_dir = os.path.join(OUTPUT_DIR, scene)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}.txt"), "w") as f:
        f.write(prompt)

def main():
    with open(PROMPT_FILE, "r") as f:
        data = json.load(f)

    with open(INDEX_FILE, "r") as f:
        index_file = json.load(f)
        image_index = index_file["image_index"] 

    items = []
    print("BUILDING INDEX...")
    for idx in tqdm(image_index):
        if idx in data:
            items.append((idx, data[idx]))
        else:
            print(f"Warning: {idx} not found in prompts.json")
        
    print("BUILDING INDEX DONE")

    with Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(save_prompt, items), total=len(items)))

if __name__ == "__main__":
    main()
