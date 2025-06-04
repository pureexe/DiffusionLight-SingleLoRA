import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool

INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024"
TARGET_DIR = "shading_exr_perspective_v3_order2_marigold"
TOTAL_SCENE = 816
NUM_PROCESSES = 8

def delete_scene(i):
    try:
        scene_name = f"{i * 1000 :06d}"
        del_dir = os.path.join(INPUT_DIR, scene_name, TARGET_DIR)
        print("DELETING... ", del_dir)
        shutil.rmtree(del_dir)
    except Exception as e:
        return f"Skipped {i}: {e}"
    return None

def main():
    with Pool(processes=NUM_PROCESSES) as pool:
        for _ in tqdm(pool.imap_unordered(delete_scene, range(TOTAL_SCENE)), total=TOTAL_SCENE):
            pass

if __name__ == "__main__":
    main()