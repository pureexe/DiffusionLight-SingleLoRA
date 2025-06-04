import os
from multiprocessing import Pool
from tqdm import tqdm

from functools import partial

def check_scene(scene_path):
    square_hdr_path = os.path.join(scene_path, "square_hdr")
    try:
        files = os.listdir(square_hdr_path)
        if len(files) != 25:
            return os.path.basename(scene_path)
    except FileNotFoundError:
        return os.path.basename(scene_path)  # Print scene if square_hdr does not exist
    return None

def main(root_dir):
    scenes = [
        os.path.join(root_dir, scene)
        for scene in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, scene))
    ]

    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(check_scene, scenes), total=len(scenes)))

    for scene in results:
        if scene:
            print(scene)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_directory>")
        exit(1)
    main(sys.argv[1])
import os
from multiprocessing import Pool
from tqdm import tqdm

from functools import partial

def check_scene(scene_path):
    square_hdr_path = os.path.join(scene_path, "square_hdr")
    try:
        files = os.listdir(square_hdr_path)
        if len(files) != 25:
            return os.path.basename(scene_path)
    except FileNotFoundError:
        return os.path.basename(scene_path)  # Print scene if square_hdr does not exist
    return None

def main(root_dir):
    scenes = [
        os.path.join(root_dir, scene)
        for scene in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, scene))
    ]

    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(check_scene, scenes), total=len(scenes)))

    for scene in results:
        if scene:
            print(scene)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_directory>")
        exit(1)
    main(sys.argv[1])
