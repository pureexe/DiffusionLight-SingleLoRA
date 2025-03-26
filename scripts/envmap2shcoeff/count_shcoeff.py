import os 
from tqdm.auto import tqdm 

TOTAL_SCENE = 816
PERSPECTIVE_PATH = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shcoeff_perspective_order100"

def main():
    count_file = 0
    for scene_id in tqdm(range(TOTAL_SCENE)):
        scene_name = f"{scene_id * 1000:06d}"
        path = PERSPECTIVE_PATH.format(scene_name)
        if not os.path.exists(path):
            continue
        files = os.listdir(path)
        count_file += len(files)
    print("TOTAL EPRSPECTIVE_ENVMAP :", count_file)
    
if __name__ == "__main__":
    main()