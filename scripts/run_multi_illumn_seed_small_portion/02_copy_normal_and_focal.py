import json 
import os 
import argparse
import shutil
from tqdm.auto import tqdm

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/train/")
    parser.add_argument("--output_dir", type=str, default="/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/")
    parser.add_argument("--scene_ids", type=str, default="scenes.json")
    parser.add_argument("--resource", type=str, default="focal,normal")
    return parser

def main():
    args = create_argparser().parse_args()

    # read scene 
    #with open(args.scene_ids, 'r') as f:
    #    scene_ids = json.load(f)
    scene_ids = sorted(os.listdir(args.input_dir))

    for scene_meta in tqdm(scene_ids):
        scene = scene_meta.split("/")[-1] 

        for resource in args.resource.split(","):
            in_dir = os.path.join(args.input_dir, scene, resource)
            out_dir = os.path.join(args.output_dir, scene_meta, resource)
            if not os.path.exists(in_dir):
                print("NOT FOUND: ", in_dir)
                continue
            if os.path.exists(out_dir):
                print("EXIST: ", out_dir)
                continue
            shutil.copytree(in_dir, out_dir, dirs_exist_ok=True)


        
    
if __name__ == "__main__":
    main()