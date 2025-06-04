import os 
from tqdm.auto import tqdm

def main():
    for seed in [100,200,300,400,500]:
        in_dir = f"/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed{seed}/"
        scenes = sorted(os.listdir(in_dir))
        total_files = 0
        for idx, scene in enumerate(tqdm(scenes)):
            # count file in scene/square_directory 
            path_dir = os.path.join(in_dir, scene, "square")
            if not os.path.exists(path_dir):
                print("NOT FOUND: ", path_dir)
                continue
            files = os.listdir(path_dir)
            total_files += len(files)
        print(f"Total files in seed {seed}: {total_files}")
        # div by 3 
        total_files //= 3
        print(f"Total files divided by 3 in seed {seed}: {total_files}")

if __name__ == "__main__":
    main()