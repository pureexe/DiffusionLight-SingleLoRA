
import os
from tqdm.auto import tqdm
IN_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/train"
OUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train"
# need to symlink norrmal and focal

def main():
    scenes = sorted(os.listdir(IN_DIR))
    for scene in tqdm(scenes):
        in_dir = os.path.join(IN_DIR, scene)
        out_dir = os.path.join(OUT_DIR, scene)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        normal_path = os.path.join(in_dir, "normal")
        focal_path = os.path.join(in_dir, "focal")
        normal_symlink_path = os.path.join(out_dir, "normal")
        focal_symlink_path = os.path.join(out_dir, "focal")
        if not os.path.exists(normal_symlink_path):
            os.symlink(normal_path, normal_symlink_path)
        if not os.path.exists(focal_symlink_path):
            os.symlink(focal_path, focal_symlink_path)

if __name__ == "__main__":
    main()