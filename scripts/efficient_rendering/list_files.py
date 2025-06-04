import os 
import json
from tqdm.auto import tqdm 

TOTAL_SCENE = 816 
COEFF_DIR_TEMPLATE = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shcoeff_perspective_v3_order100"
def efficient_rendering():    
    queues = []
    print("READING SCENE...")
    scene_lists = list(range(TOTAL_SCENE))
    for scene_id in tqdm(scene_lists):
        scene_name = f"{scene_id*1000:06d}"
        coeff_dir = COEFF_DIR_TEMPLATE.format(scene_name)
        if not os.path.exists(coeff_dir):
            continue
        filenames = sorted([a for a in os.listdir(coeff_dir) if a.endswith('.npy')])
        for filename in filenames:
            queues.append([scene_name, filename.replace('.npy', '')])
    print("SAVING files...")
    with open('queues.json','w') as f:
        json.dump(queues,f)
    print("DONE")

if __name__ == "__main__":
    efficient_rendering()