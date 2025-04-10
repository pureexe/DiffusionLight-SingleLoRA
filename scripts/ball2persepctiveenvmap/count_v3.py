import os
from tqdm.auto import tqdm 

TOTAL_SCENE = 816
TEMPLATE = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/envmap_perspective_v3"

def main():
    
    count_file = 0
    
    for scene_id in tqdm(range(TOTAL_SCENE)):
        scene_name = f"{scene_id*1000:06d}"
        image_dir = TEMPLATE.format(scene_name)
        try:
            image_files = os.listdir(image_dir)
        except:
            continue
        images_files = [f for f in image_files if f.endswith(".exr")]
        count_file += len(images_files)
        
    print(f"Total number of files: {count_file}")
        
    

if __name__ == "__main__":
    main()