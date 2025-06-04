import os 
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    parser.add_argument("-t", "--total", type=int, default=1)
    return parser


SEEDS = [100, 200, 300, 400, 500]
# SCENES = [
#     "14n_copyroom1",
#     "14n_copyroom10",
#     "14n_copyroom6",
#     "14n_copyroom8",
#     "14n_office12"
# ]

INPUT_DIR = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed{seed}/{scene}/square"
OUTPUT_DIR = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed{seed}/{scene}/square_hdr"

def main():
    SCENES = sorted(os.listdir("/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100/"))
    args = create_argparser().parse_args()
    for seed in SEEDS:
        for idx, scene in enumerate(SCENES):
            print(f"Processing seed {seed}, scene {scene} ({idx+1}/{len(SCENES)})")
            in_dir = INPUT_DIR.format(seed=seed, scene=scene)  # Just to get the base path
            out_dir = OUTPUT_DIR.format(seed=seed, scene=scene)
            if not os.path.exists(in_dir):
                print("NOT FOUND: ", in_dir)
                continue
            cmd = f"python exposure2hdr.py --input_dir {in_dir} --output_dir {out_dir}"
            os.system(cmd)

        
    
if __name__ == "__main__":
    main()