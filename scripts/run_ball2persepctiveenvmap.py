import os 
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    parser.add_argument("--threads", type=int, default=20)
    return parser

TOTAL_SCENE = 816

INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/square_hdr"
FOCAL_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/focal"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/envmap_perspective_fov"

def main():
    args = create_argparser().parse_args()
    scene_ids = range(408,TOTAL_SCENE)
    scene_ids = scene_ids[args.idx::args.total]
    for scene_id in scene_ids:
        scene_name = f"{scene_id*1000:06d}"
        in_dir = INPUT_DIR.format(scene_name)
        focal_dir = FOCAL_DIR.format(scene_name)
        out_dir = OUTPUT_DIR.format(scene_name)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        cmd = f"python ball2perspectiveenvmap.py --ball_dir {in_dir} --envmap_dir {out_dir} --fov_dir {focal_dir} --threads {args.threads}"
        print(cmd)
        os.system(cmd)

    
if __name__ == "__main__":
    main()
    
# Total files ending with "_ev-00.png": 257054