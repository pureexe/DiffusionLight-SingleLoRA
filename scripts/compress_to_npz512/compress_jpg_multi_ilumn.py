import os
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

# INPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_render_from_fitting_v2"
# OUTPUT_DIR = "/data2/pakkapon/image512_lstsq"
INPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images"
OUTPUT_DIR = "/data2/pakkapon/image512_real"
NUM_WORKERS = 40

def convert_to_jpg(task):
    input_path, output_path = task
    try:
        # Open and convert image
        img = Image.open(input_path).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path.with_suffix('.jpg'), "JPEG")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def gather_tasks(input_root, output_root):
    print("Gathering tasks...")
    tasks = []
    for scene in tqdm(os.listdir(input_root)):
        scene_path = Path(input_root) / scene
        if scene_path.is_dir():
            for file in os.listdir(scene_path):
                input_file = scene_path / file
                if input_file.is_file():
                    output_file = Path(output_root) / scene / input_file.stem
                    tasks.append((input_file, output_file))
    print(f"Total tasks: {len(tasks)}")
    return tasks

if __name__ == "__main__":
    tasks = gather_tasks(INPUT_DIR, OUTPUT_DIR)

    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(convert_to_jpg, tasks), total=len(tasks)))
