import os 
from tqdm.auto import tqdm

def main():
    IMAGE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images"
    scenes = sorted(os.listdir(IMAGE_DIR))
    total_files = 0
    for idx, scene in enumerate(tqdm(scenes)):
        # count file in scene/square_directory 
        path_dir = os.path.join(IMAGE_DIR, scene, )
        if not os.path.exists(path_dir):
            print("NOT FOUND: ", path_dir)
            continue
        files = os.listdir(path_dir)
        total_files += len(files)
    print(f"Total files in dataset: {total_files}")

if __name__ == "__main__":
    main()