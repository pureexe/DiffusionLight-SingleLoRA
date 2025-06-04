import os
from tqdm.auto import tqdm
BASE_DIR = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination"
SEEDS = [200, 300, 400, 500]
SOURCE_SEED = 100

source_root = os.path.join(BASE_DIR, f"seed{SOURCE_SEED}")
scene_names = [name for name in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, name))]

for scene in tqdm(scene_names):
    source_scene_path = os.path.join(source_root, scene)
    
    for subdir in ["focal", "normal"]:
        source_subdir = os.path.join(source_scene_path, subdir)
        if not os.path.exists(source_subdir):
            print(f"Warning: {source_subdir} does not exist. Skipping.")
            continue

        for seed in SEEDS:
            target_scene_path = os.path.join(BASE_DIR, f"seed{seed}", scene)
            target_symlink = os.path.join(target_scene_path, subdir)

            os.makedirs(target_scene_path, exist_ok=True)

            if os.path.islink(target_symlink) or os.path.exists(target_symlink):
                print(f"Symlink or file already exists at {target_symlink}. Skipping.")
                continue

            os.symlink(source_subdir, target_symlink)
            print(f"Created symlink: {target_symlink} -> {source_subdir}")