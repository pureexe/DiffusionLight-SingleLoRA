import ezexr 
import os 
from tqdm.auto import tqdm


IN_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen2/shading_exr_perspective_v3_order6_ball"

def main():
    files = os.listdir(IN_DIR)
    files = sorted(files)
    min_value = 1e10
    for filename in tqdm(files):
        if filename.endswith(".exr"):
            in_path = os.path.join(IN_DIR, filename)
            image = ezexr.imread(in_path) 
            min_value = min(min_value, image.min())

    print("MIN_VALUE: ", min_value)

if __name__ == "__main__":
    main()