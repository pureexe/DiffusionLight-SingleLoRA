import os 
import argparse 

HUGGINGFACE_PATH="/ist/ist-share/vision/huggingface"
DATASET_PATH = "/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images"
OUTPUT_PATH = "output/laion-aesthetics-1024/"
CACHE_DIR = "output/cache/laion-aesthetics-1024/"

"""
CUDA_VISIBLE_DEVICES=2 python validate_2lora.py \
    --dataset $dataset \
    --output_dir $output_dir/$config/$train_steps_format \
    --cache_dir $output_dir/$config/$train_steps_format/cache \
    --no_torch_compile \
    --ev "0,-2.5,-5" \
    --denoising_step 30 \
    --model_option sdxl \
    --control_scale 0.5 \
    --lora_path ./real_checkpoint/$config/checkpoint-$train_steps \
    --lora_scale 1.0 \
    --guidance_scale 5 \
    --algorithm special \
    --no_random_loader \
"""

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--num_dir", type=int, default=816) 
    return parser

def main():
    args = create_argparser().parse_args()
    total_scene = ((args.num_dir - 1) // args.total) + 1
    for dir_id in range(args.idx * total_scene, (args.idx+1) * total_scene):
        # make a scrash
        dirname = f"{(dir_id * 1000):06d}"
        dataset_path = os.path.join(DATASET_PATH, dirname)
        output_path = os.path.join(OUTPUT_PATH, dirname)
        cache_path = os.path.join(CACHE_DIR, dirname)
        cmd = f'HF_DATASETS_CACHE="{HUGGINGFACE_PATH}" python validate_2lora.py --dataset {dataset_path} --cache_dir {cache_path}  --output_dir {output_path} --no_save_intermediate --no_torch_compile --no_random_loader --lora_path real_checkpoint/rev3/Flickr2K/Flickr2K_balanced_aligned/checkpoint-140000'
        print("_____ COMMAND _____")
        print(cmd)
        print("___________________")
        os.system(cmd)
    
if __name__ == "__main__":
    main()