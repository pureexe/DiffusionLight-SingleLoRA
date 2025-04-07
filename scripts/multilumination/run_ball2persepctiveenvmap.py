import os 
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    parser.add_argument("--threads", type=int, default=25)
    return parser

TOTAL_SCENE = 816

SPLIT_TYPE = 'train'
INPUT_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT_TYPE}/square_hdr_gt"
FOCAL_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT_TYPE}/metric_focallength"
OUTPUT_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT_TYPE}/envmap_perspective"

def main():
    args = create_argparser().parse_args()
    scene_ids = sorted(os.listdir(INPUT_DIR))
    for scene_name in scene_ids:
        in_dir = os.path.join(INPUT_DIR, scene_name)
        focal_dir = os.path.join(FOCAL_DIR, scene_name)
        out_dir = os.path.join(OUTPUT_DIR, scene_name)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        cmd = f"python ball2perspectiveenvmap.py --ball_dir {in_dir} --envmap_dir {out_dir} --fov_dir {focal_dir} --threads {args.threads}"
        print(cmd)
        os.system(cmd)

    
if __name__ == "__main__":
    main()
    
# Total files ending with "_ev-00.png": 257054