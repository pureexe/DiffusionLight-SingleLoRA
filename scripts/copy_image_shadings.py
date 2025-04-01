# copy image shadings 

import os 
import shutil
from tqdm.auto import tqdm
from multiprocessing import Pool 
import ezexr
import skimage
import json

THREAD = 40
TOTAL_SCENE = 816
INPUT_IMAGE_DIR = "/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images/{}"
INPUT_SHADING_DIR = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{}/shading_exr_perspective_fov_order6"
OUTPUT_DIR = "/ist/ist-share/vision/relight/datasets/laion-shading/v1/train"
DEFAULT_PROMPT = "a photorealistic image"

def process_scene(scene_id):
    scene_name = f"{scene_id * 1000:06d}"
    input_image_dir = INPUT_IMAGE_DIR.format(scene_name)
    input_shading_dir = INPUT_SHADING_DIR.format(scene_name)
    image_files = [a.replace('.jpg','') for a in os.listdir(input_image_dir)]
    shading_files = [a.replace('.jpg','') for a in os.listdir(input_image_dir)]
    # find intersect 
    files = list(set(image_files) & set(shading_files))
    files = sorted(files)
    outputs = []
    output_image_dir = os.path.join(OUTPUT_DIR, "images", scene_name)
    os.makedirs(output_image_dir, exist_ok=True)
    os.chmod(output_image_dir, 0o777)
    output_shading_dir = os.path.join(OUTPUT_DIR, "shadings", scene_name)
    os.makedirs(output_shading_dir, exist_ok=True)
    os.chmod(output_shading_dir, 0o777)
    
    # print(f"copying image... {scene_name}")
    # shutil.copytree(input_image_dir, output_image_dir, dirs_exist_ok=True)
    # print(f"copying shading... {scene_name}")
    # shutil.copytree(input_shading_dir, output_shading_dir, dirs_exist_ok=True)
    # print(f"chmoding.. {scene_name}")

    for fname in files:
        try:
            input_image_path = os.path.join(input_image_dir, fname+".jpg")
            output_image_path = os.path.join(output_image_dir, fname+".jpg")
            input_shading_path = os.path.join(input_shading_dir, fname+".exr")
            output_shading_path = os.path.join(output_shading_dir, fname+".exr")
            if os.path.exists(output_image_path) and os.path.exists(output_shading_path):
                outputs.append("f{scene_name}/{fname}")
                continue
            
            if not os.path.exists(input_image_path):
                continue
            if not os.path.exists(input_shading_path):
                continue
            
            # test read image, if failed, we skip
            # try:
            #     test_read = skimage.io.imread(input_image_path)
            # except:
            #     continue
            # try:
            #     test_read = ezexr.imread(input_shading_path)
            # except:
            #     continue

            if not os.path.exists(output_image_path):
                shutil.copy2(input_image_path, output_image_path)
                os.chmod(output_image_path, 0o777)
            if not os.path.exists(output_shading_path):
                shutil.copy2(input_shading_path, output_shading_path)
                os.chmod(output_shading_path, 0o777)
            outputs.append("f{scene_name}/{fname}")
        except:
            continue
    return outputs


def main():
    #process_scene(0)
    START_FRAME = 408
    with Pool(THREAD) as p:
        results = list(tqdm(p.imap(process_scene, range(START_FRAME, TOTAL_SCENE)), total=len(range(START_FRAME, TOTAL_SCENE))))
        exit()
        indexs = {
            "image_index": [],
            "envmap_index": []
        }
        prompts = {}
        for scene in results:
            indexs["image_index"] += scene
            for filename in scene:
                indexs['envmap_index'].append([
                    filename
                ])
                prompts[filename] = DEFAULT_PROMPT
        # save json 
        with open("data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        print("TOTAL FILES: ", len(indexs['image_index']))
    
if __name__ == "__main__":
    main()