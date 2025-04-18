import os 
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    return parser

TOTAL_SCENE = 816

INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/square"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/square_hdr"

def main():
    args = create_argparser().parse_args()
    scene_ids = range(0,TOTAL_SCENE)
    scene_ids = scene_ids[args.idx::args.total]
    for scene_id in scene_ids:
        scene_name = f"{scene_id*1000:06d}"
        in_dir = INPUT_DIR.format(scene_name)
        out_dir = OUTPUT_DIR.format(scene_name)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        cmd = f"python exposure2hdr.py --input_dir {in_dir} --output_dir {out_dir}"
        os.system(cmd)

        
    
if __name__ == "__main__":
    main()