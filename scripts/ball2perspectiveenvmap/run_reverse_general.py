import os 
import argparse
import json

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--square_hdr_dir", type=str, default="square_hdr")
    parser.add_argument("--focal_dir", type=str, default="focal")
    parser.add_argument("--input_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train")
    parser.add_argument("--output_dir", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train")
    parser.add_argument("--envmap_dir", type=str, default="envmap_perspective_v3")
    parser.add_argument("--scene_ids", type=str, default="")
    return parser


def main():
    args = create_argparser().parse_args()
    if args.scene_ids != "":
        with open(args.scene_ids, 'r') as f:
            scene_ids = json.load(f)
    else:
        print(os.listdir("/pure"))
        scene_ids = sorted(os.listdir(args.input_dir))
    scene_ids = scene_ids[args.idx::args.total]
    for scene_name in scene_ids:
        in_dir = os.path.join(args.input_dir, scene_name, args.square_hdr_dir)
        focal_dir = os.path.join(args.input_dir, scene_name, args.focal_dir)
        out_dir = os.path.join(args.output_dir, scene_name, args.envmap_dir)
        if not os.path.exists(in_dir):
            print("NOT FOUND: ", in_dir)
            continue
        # counte number of file 
        if os.path.exists(out_dir):
            in_dir_count = len(os.listdir(in_dir))
            out_dir_count = len(os.listdir(out_dir))
            # file nmumber or already equal
            if in_dir_count == out_dir_count and in_dir_count == 25: 
                continue
        print("==================================================================")
        print("Processing scene: ", scene_name)
        cmd = f"python reverse_wrapping.py --ball_dir {in_dir} --envmap_dir {out_dir} --focal_dir {focal_dir} --threads {args.threads}"
        print(cmd)
        os.system(cmd)

    
if __name__ == "__main__":
    main()
    
# Total files ending with "_ev-00.png": 257054