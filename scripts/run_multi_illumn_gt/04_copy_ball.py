import os
from tqdm.auto import tqdm
import shutil
from multiprocessing import Pool, cpu_count

INPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination_test_mip2_exr"
OUTPUT_DIR = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/test"

def process_scene(scene):
    scene_path = os.path.join(INPUT_DIR, scene)
    if not os.path.exists(scene_path):
        print(f"Scene path does not exist: {scene_path}")
        return
    
    try:
        files = sorted(os.listdir(os.path.join(scene_path, "probes")))
        probes = [a for a in files if a.endswith("_chrome256.exr")]
        for probe in probes:
            incorrect_probe_dir= os.path.join(OUTPUT_DIR, scene, "probes")
            if os.path.exists(incorrect_probe_dir):
                shutil.rmtree(incorrect_probe_dir)
            target_probe_path = os.path.join(OUTPUT_DIR, scene, "square_hdr", probe.replace("_chrome256.exr", "_mip2.exr"))
            source_probe_path = os.path.join(scene_path, "probes", probe)
            os.makedirs(os.path.dirname(target_probe_path), exist_ok=True)
            shutil.copyfile(source_probe_path, target_probe_path)
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def main():
    scenes = sorted(os.listdir(INPUT_DIR))
    scenes = [scene for scene in scenes if os.path.isdir(os.path.join(INPUT_DIR, scene))]

    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_scene, scenes), total=len(scenes)))

if __name__ == "__main__":
    main()
