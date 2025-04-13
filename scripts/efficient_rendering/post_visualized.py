INPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/shading_exr_perspective_v3_order2"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/shading_exr_perspective_v3_order2_viz_ldr"

import os
from tonemapper import TonemapHDR
from tqdm.auto import tqdm 
from multiprocessing import Pool 
import ezexr
import skimage

def process_scene(filename):
    in_path = os.path.join(INPUT_DIR, filename)
    viz_path = os.path.join(OUTPUT_DIR, filename.replace(".exr", ".png"))
    # read exr file
    image = ezexr.imread(in_path)
    # convert to tonemapper
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    tonemapped_shading, _, _ = tonemap(image)
    skimage.io.imsave(viz_path, skimage.img_as_ubyte(tonemapped_shading))
    os.chmod(viz_path, 0o777)

def main():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chmod(OUTPUT_DIR, 0o777)

    # list all file in INPUT_DIR
    files = sorted(os.listdir(INPUT_DIR))
    # filter only exr file 
    exr_files = [f for f in files if f.endswith(".exr")]
    
    # process using multi-threading
    with Pool(processes=8) as pool:
        # use tqdm with imap 
        r = list(tqdm(pool.imap(process_scene, exr_files), total=len(exr_files)))
    
if __name__ == "__main__":
    main()