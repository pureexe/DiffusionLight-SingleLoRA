import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from skimage.io import imread
from skimage import io
import warnings

# Suppress warnings from skimage for corrupted files
warnings.filterwarnings("ignore", category=UserWarning)

# Change this to your root directory containing scene folders
ROOT_PATH = '/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images'

def validate_image(image_path):
    try:
        _ = imread(image_path)
        return (image_path, True)
    except Exception:
        return (image_path, False)

def process_scene(scene_path):
    results = []
    scene_dir = os.path.join(ROOT_PATH, scene_path)
    if not os.path.isdir(scene_dir):
        return results
    for image_name in os.listdir(scene_dir):
        image_path = os.path.join(scene_dir, image_name)
        if os.path.isfile(image_path):
            results.append(image_path)
    return results

def main():
    scenes = [s for s in os.listdir(ROOT_PATH) if os.path.isdir(os.path.join(ROOT_PATH, s))]
    #scenes = ["000000"]
    # Collect all image paths
    all_images = []
    for scene in tqdm(scenes):
        image_paths = process_scene(scene)
        all_images.extend(image_paths)

    print(f"Found {len(all_images)} images to validate.")

    with Pool(processes=40) as pool:
        for image_path, success in tqdm(pool.imap_unordered(validate_image, all_images), total=len(all_images)):
            if not success:
                try:
                    os.remove(image_path)
                    print(f"Deleted corrupted image: {image_path}")
                except Exception as e:
                    print(f"Failed to delete {image_path}: {e}")

if __name__ == "__main__":
    main()
