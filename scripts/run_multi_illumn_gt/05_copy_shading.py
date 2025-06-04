import os 
from tqdm.auto import tqdm
import shutil
from multiprocessing import Pool

INPUT_DIR = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train"
OUTPUT_DIR = "/pure/f1/datasets/multi_illumination/real_image_gt_shading/v0/train/shadings"

def copy_shading(scene):
    input_dir = os.path.join(INPUT_DIR, scene, "shading_exr_perspective_v3_order6_v2")
    output_dir = os.path.join(OUTPUT_DIR, scene)
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

def main():
    scenes = sorted(os.listdir(INPUT_DIR))
    with Pool(24) as p:
        r = list(tqdm(p.imap(copy_shading, scenes), total=len(scenes)))
    
    

if __name__ == "__main__":
    main()