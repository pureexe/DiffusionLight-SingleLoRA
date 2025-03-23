import os 
import argparse 

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    os.system(f"singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/  --env HF_HOME=/ist/users/pakkaponp/.cache/huggingface /ist/ist-share/vision/pakkapon/singularity/diffusionlight0230.sif python command/runner.py --idx {args.idx} --total {args.total}")