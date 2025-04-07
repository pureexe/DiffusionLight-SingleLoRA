from tqdm.auto import tqdm 
import os 
from multiprocessing import Pool
import shutil
THREAD = 8
SPLITTYPE = 'test'
INPUT_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination_{SPLITTYPE}_mip2_exr"
CHROMEBALL_IN = "probes/dir_{}_chrome256.exr"
OUTPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/square_hdr_gt"
CHROMEBALL_OUT = "dir_{}_mip2.exr"

def process_image(args):
    chromeball_in, out_dir, out_name = args
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return out_path
    # Copy the file
    shutil.copy(chromeball_in, out_path) 
    return out_path    

def main():
    scenes = os.listdir(INPUT_DIR)
    queues = []
    for scene in scenes:
        scene_dir = os.path.join(INPUT_DIR, scene)
        if not os.path.isdir(scene_dir):
            continue
        for light in range(25):
            chromeball_in = os.path.join(scene_dir, CHROMEBALL_IN.format(light))
            out_dir = os.path.join(OUTPUT_DIR, scene)
            out_name = CHROMEBALL_OUT.format(light)
            queues.append((chromeball_in, out_dir, out_name))
                
    with Pool(THREAD) as p:
        list(tqdm(p.imap(process_image, queues), total=len(queues)))

    
if __name__ == "__main__":
    main()