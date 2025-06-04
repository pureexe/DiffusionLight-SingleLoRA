import os 
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    return parser


INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train/{}/square"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train/{}/square_hdr"

def main():
    args = create_argparser().parse_args()
    input_dir = INPUT_DIR.split('/{}/')[0]
    scene_ids = sorted(os.listdir(input_dir))
    #scene_ids = ['elm_2floor_bathroom2']
    scene_ids = scene_ids[args.idx::args.total]
    for scene_name in scene_ids:
        in_dir = INPUT_DIR.format(scene_name)
        out_dir = OUTPUT_DIR.format(scene_name)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        cmd = f"python exposure2hdr.py --input_dir {in_dir} --output_dir {out_dir}"
        os.system(cmd)

        
    
if __name__ == "__main__":
    main()