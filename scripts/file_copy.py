import os 
import shutil
from tqdm.auto import tqdm
TOTAL_SCENE = 816
INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024"
OUTPUT_DIR = "/pure/t1/output/DiffusionLight-SingleLoRA/laion-aesthetics-1024"
TO_COPY_DIRS = ["raw"]

def main():
    print("START COPYING PROCESS")
    for i in tqdm(range(TOTAL_SCENE)):
        name = f"{i*1000:06d}"
        if not os.path.exists(os.path.join(INPUT_DIR, name)):
            continue
        output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(output_dir, exist_ok=True)
        try:
            os.chmod(output_dir, 0o777)
        except:
            pass
        for current_dir in TO_COPY_DIRS:
            in_path = os.path.join(INPUT_DIR, name, current_dir)
            if not os.path.exists(in_path):
                continue
            out_path = os.path.join(OUTPUT_DIR,name, current_dir)
            os.system(f'rclone copy {in_path} {out_path} --transfers 8 --ignore-existing')
            os.chmod(in_path, 0o777)    


if __name__ == "__main__":
    main()